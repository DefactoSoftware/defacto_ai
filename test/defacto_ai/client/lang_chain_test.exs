defmodule DefactoAI.Client.LangChainTest do
  @moduledoc """
  Integration tests for the production `DefactoAI.Client.LangChain`
  implementation. Spins up a Bypass server that mimics OpenAI's chat
  completions endpoint and exercises:

    * the tool-call happy path
    * inter-strategy fallback when the provider rejects tool calling
    * intra-strategy validation retry (the repair loop)
    * the JSON repair pipeline against messy text-mode output
    * plain chat completion
    * the streaming chat path (including non-standard SSE shapes)

  Routes are queue-shaped (Bypass.expect with a counter) so each
  scenario can declare the expected sequence of upstream calls.
  """

  use ExUnit.Case, async: false

  alias DefactoAI.Client.LangChain
  alias DefactoAI.TestProvider

  setup do
    bypass = Bypass.open()
    {:ok, bypass: bypass, base_url: "http://localhost:#{bypass.port}"}
  end

  defmodule TestSchema do
    use Ecto.Schema
    import Ecto.Changeset

    @primary_key false
    embedded_schema do
      field(:answer, :string)
    end

    def changeset(struct, attrs, _opts \\ []) do
      struct
      |> cast(attrs, [:answer])
      |> validate_required([:answer])
    end

    def parameters_schema do
      %{
        type: "object",
        properties: %{answer: %{type: "string", description: "The answer."}},
        required: ["answer"],
        additionalProperties: false
      }
    end
  end

  defp provider(base_url) do
    TestProvider.new(%{base_url: base_url, api_path: "/v1/chat/completions"})
  end

  defp messages, do: [%{role: "user", content: "what's the answer?"}]

  defp openai_tool_call_response(answer) do
    args =
      case answer do
        a when is_binary(a) -> %{answer: a}
        :empty -> %{}
      end

    Jason.encode!(%{
      id: "chatcmpl-test",
      object: "chat.completion",
      created: 1,
      model: "gpt-test",
      choices: [
        %{
          index: 0,
          message: %{
            role: "assistant",
            content: nil,
            tool_calls: [
              %{
                id: "call_1",
                type: "function",
                function: %{name: "respond", arguments: Jason.encode!(args)}
              }
            ]
          },
          finish_reason: "tool_calls"
        }
      ],
      usage: %{prompt_tokens: 1, completion_tokens: 1, total_tokens: 2}
    })
  end

  defp openai_text_response(content) do
    Jason.encode!(%{
      id: "chatcmpl-test",
      object: "chat.completion",
      created: 1,
      model: "gpt-test",
      choices: [
        %{
          index: 0,
          message: %{role: "assistant", content: content},
          finish_reason: "stop"
        }
      ],
      usage: %{prompt_tokens: 1, completion_tokens: 1, total_tokens: 2}
    })
  end

  defp expect_in_order(bypass, responses) do
    counter = :atomics.new(1, [])

    Bypass.expect(bypass, "POST", "/v1/chat/completions", fn conn ->
      idx = :atomics.add_get(counter, 1, 1) - 1
      {status, body, content_type} = Enum.at(responses, idx)

      conn
      |> Plug.Conn.put_resp_content_type(content_type)
      |> Plug.Conn.resp(status, body)
    end)
  end

  defp ok(body), do: {200, body, "application/json"}

  defp api_error(status, message) do
    body = Jason.encode!(%{error: %{message: message, type: "invalid_request_error"}})
    {status, body, "application/json"}
  end

  describe "complete_structured/3" do
    test "returns a struct from the parsed tool-call arguments", %{bypass: bypass, base_url: url} do
      expect_in_order(bypass, [ok(openai_tool_call_response("42"))])

      assert {:ok, %TestSchema{answer: "42"}} =
               LangChain.complete_structured(TestSchema, messages(), provider: provider(url))
    end

    test "errors when no provider given and no resolver configured" do
      previous = Application.get_env(:defacto_ai, :provider_resolver)
      Application.delete_env(:defacto_ai, :provider_resolver)

      try do
        assert {:error, :no_provider_resolver} =
                 LangChain.complete_structured(TestSchema, messages(), [])
      after
        if previous, do: Application.put_env(:defacto_ai, :provider_resolver, previous)
      end
    end

    test "uses the provider_resolver callback when no explicit provider given",
         %{bypass: bypass, base_url: url} do
      expect_in_order(bypass, [ok(openai_tool_call_response("via-resolver"))])

      previous = Application.get_env(:defacto_ai, :provider_resolver)

      Application.put_env(:defacto_ai, :provider_resolver, fn :llm -> provider(url) end)

      try do
        assert {:ok, %TestSchema{answer: "via-resolver"}} =
                 LangChain.complete_structured(TestSchema, messages(), [])
      after
        if previous,
          do: Application.put_env(:defacto_ai, :provider_resolver, previous),
          else: Application.delete_env(:defacto_ai, :provider_resolver)
      end
    end

    test "errors {:no_provider_for_role, role} when resolver returns nil" do
      previous = Application.get_env(:defacto_ai, :provider_resolver)
      Application.put_env(:defacto_ai, :provider_resolver, fn _ -> nil end)

      try do
        assert {:error, {:no_provider_for_role, :llm}} =
                 LangChain.complete_structured(TestSchema, messages(), [])
      after
        if previous,
          do: Application.put_env(:defacto_ai, :provider_resolver, previous),
          else: Application.delete_env(:defacto_ai, :provider_resolver)
      end
    end

    test "falls back to JSON mode when the provider rejects tool_choice",
         %{bypass: bypass, base_url: url} do
      expect_in_order(bypass, [
        api_error(400, "tool_choice is not supported by this model"),
        ok(openai_text_response(~s({"answer": "fallback"})))
      ])

      assert {:ok, %TestSchema{answer: "fallback"}} =
               LangChain.complete_structured(TestSchema, messages(), provider: provider(url))
    end

    test "falls all the way through to text-repair when both tool and JSON modes fail",
         %{bypass: bypass, base_url: url} do
      messy = """
      Sure! Here you go:

      ```json
      {"answer": "all-the-way",}
      ```
      """

      expect_in_order(bypass, [
        api_error(400, "tool_choice not supported"),
        api_error(400, "response_format json_object not supported"),
        ok(openai_text_response(messy))
      ])

      assert {:ok, %TestSchema{answer: "all-the-way"}} =
               LangChain.complete_structured(TestSchema, messages(), provider: provider(url))
    end

    test "appends a corrective message and retries on validation failure",
         %{bypass: bypass, base_url: url} do
      expect_in_order(bypass, [
        ok(openai_tool_call_response(:empty)),
        ok(openai_tool_call_response("second-try"))
      ])

      assert {:ok, %TestSchema{answer: "second-try"}} =
               LangChain.complete_structured(TestSchema, messages(), provider: provider(url))
    end

    test "exhausts the retry budget and surfaces a validation error",
         %{bypass: bypass, base_url: url} do
      # budget = 2 means: initial call + 2 retries = 3 upstream calls
      expect_in_order(bypass, [
        ok(openai_tool_call_response(:empty)),
        ok(openai_tool_call_response(:empty)),
        ok(openai_tool_call_response(:empty))
      ])

      assert {:error, {:validation_failed, %Ecto.Changeset{valid?: false}}} =
               LangChain.complete_structured(TestSchema, messages(),
                 provider: provider(url),
                 max_validation_retries: 2
               )
    end
  end

  describe "complete_chat/2" do
    test "returns the assembled assistant content as a binary",
         %{bypass: bypass, base_url: url} do
      expect_in_order(bypass, [ok(openai_text_response("Hello there!"))])

      assert {:ok, "Hello there!"} =
               LangChain.complete_chat(
                 [%{role: "user", content: "hi"}],
                 provider: provider(url)
               )
    end

    test "classifies API errors", %{bypass: bypass, base_url: url} do
      expect_in_order(bypass, [api_error(500, "internal server error")])

      assert {:error, {:api_error, _}} =
               LangChain.complete_chat(
                 [%{role: "user", content: "hi"}],
                 provider: provider(url)
               )
    end
  end

  describe "stream_chat/2" do
    test "yields content chunks from a streamed response",
         %{bypass: bypass, base_url: url} do
      sse_route(bypass, [chunk_json("Hello, "), chunk_json("world!")])

      assert {:ok, stream} =
               LangChain.stream_chat(
                 [%{role: "user", content: "hi"}],
                 provider: provider(url)
               )

      assert Enum.to_list(stream) == ["Hello, ", "world!"]
    end

    test "tolerates non-standard chunks missing index/finish_reason fields",
         %{bypass: bypass, base_url: url} do
      sse_route(bypass, [
        ~s({"choices":[{"delta":{"content":"works "}}]}),
        ~s({"choices":[{"delta":{"content":"anyway"}}]})
      ])

      assert {:ok, stream} =
               LangChain.stream_chat(
                 [%{role: "user", content: "x"}],
                 provider: provider(url)
               )

      assert Enum.to_list(stream) == ["works ", "anyway"]
    end

    test "halts after yielding an upstream error", %{bypass: bypass, base_url: url} do
      Bypass.expect_once(bypass, "POST", "/v1/chat/completions", fn conn ->
        Plug.Conn.resp(conn, 500, Jason.encode!(%{error: %{message: "boom"}}))
      end)

      assert {:ok, stream} =
               LangChain.stream_chat(
                 [%{role: "user", content: "x"}],
                 provider: provider(url)
               )

      # Take more than we expect to confirm the stream halts on its own
      assert [{:error, {:api_error, 500, _}}] = Enum.take(stream, 5)
    end

    test "drops chunks that don't contain content", %{bypass: bypass, base_url: url} do
      sse_route(bypass, [
        ~s({"choices":[{"delta":{"role":"assistant"}}]}),
        chunk_json("hello"),
        ~s({"choices":[{"delta":{}}]})
      ])

      assert {:ok, stream} =
               LangChain.stream_chat(
                 [%{role: "user", content: "x"}],
                 provider: provider(url)
               )

      assert Enum.to_list(stream) == ["hello"]
    end
  end

  describe "complete_structured/3 with streaming enabled" do
    setup do
      Application.put_env(:defacto_ai, :chat_stream, true)
      on_exit(fn -> Application.delete_env(:defacto_ai, :chat_stream) end)
      :ok
    end

    test "assembles streamed tool-call argument deltas into a struct",
         %{bypass: bypass, base_url: url} do
      sse_route(bypass, [
        tool_call_open("respond", "call_1"),
        tool_call_args(~s({"answer":)),
        tool_call_args(~s("42"}))
      ])

      assert {:ok, %TestSchema{answer: "42"}} =
               LangChain.complete_structured(TestSchema, messages(), provider: provider(url))
    end

    test "tolerates tool-call chunks missing index/id fields",
         %{bypass: bypass, base_url: url} do
      sse_route(bypass, [
        ~s({"choices":[{"delta":{"tool_calls":[{"function":{"name":"respond","arguments":"{\\"answer\\":\\"x\\"}"}}]}}]})
      ])

      assert {:ok, %TestSchema{answer: "x"}} =
               LangChain.complete_structured(TestSchema, messages(), provider: provider(url))
    end

    test "assembles streamed content for the JSON-mode strategy",
         %{bypass: bypass, base_url: url} do
      sse_route(bypass, [chunk_json(~s({"answer":)), chunk_json(~s("hi"}))])

      assert {:ok, %TestSchema{answer: "hi"}} =
               LangChain.complete_structured(TestSchema, messages(),
                 provider: provider(url),
                 strategies: [DefactoAI.Strategy.JsonMode]
               )
    end
  end

  defp tool_call_open(name, id) do
    Jason.encode!(%{
      choices: [
        %{
          index: 0,
          delta: %{
            role: "assistant",
            tool_calls: [
              %{index: 0, id: id, type: "function", function: %{name: name, arguments: ""}}
            ]
          }
        }
      ]
    })
  end

  defp tool_call_args(arguments) do
    Jason.encode!(%{
      choices: [
        %{index: 0, delta: %{tool_calls: [%{index: 0, function: %{arguments: arguments}}]}}
      ]
    })
  end

  defp chunk_json(content) do
    Jason.encode!(%{
      id: "chatcmpl-test",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-test",
      choices: [
        %{index: 0, delta: %{role: "assistant", content: content}, finish_reason: nil}
      ]
    })
  end

  defp sse_route(bypass, json_chunks) do
    body =
      json_chunks
      |> Enum.map(&"data: #{&1}\n\n")
      |> Enum.join()
      |> Kernel.<>("data: [DONE]\n\n")

    Bypass.expect(bypass, "POST", "/v1/chat/completions", fn conn ->
      conn
      |> Plug.Conn.put_resp_content_type("text/event-stream")
      |> Plug.Conn.send_resp(200, body)
    end)
  end
end
