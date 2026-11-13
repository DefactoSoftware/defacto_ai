defmodule DefactoAI.UsageTelemetryTest do
  @moduledoc """
  Stop-event metadata (provider, model, role, usage, result, tags) for
  every call type, exercised through the `DefactoAI.Client` and
  `DefactoAI.Embeddings` facades against a Bypass server that speaks the
  OpenAI wire format.
  """

  use ExUnit.Case, async: false

  alias DefactoAI.Client
  alias DefactoAI.Embeddings
  alias DefactoAI.TestProvider

  @events [
    [:defacto_ai, :complete_structured],
    [:defacto_ai, :complete_chat],
    [:defacto_ai, :stream_chat],
    [:defacto_ai, :embed],
    [:defacto_ai, :embed_batch]
  ]

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

  def forward(event, measurements, metadata, test_pid),
    do: send(test_pid, {:telemetry, event, measurements, metadata})

  setup do
    bypass = Bypass.open()
    test_pid = self()
    handler_id = "usage-telemetry-#{inspect(make_ref())}"

    events = for base <- @events, suffix <- [:start, :stop, :exception], do: base ++ [suffix]

    :telemetry.attach_many(handler_id, events, &__MODULE__.forward/4, test_pid)

    on_exit(fn -> :telemetry.detach(handler_id) end)

    provider =
      TestProvider.new(%{
        base_url: "http://localhost:#{bypass.port}",
        provider_type: "acme",
        model: "model-1"
      })

    {:ok, bypass: bypass, provider: provider}
  end

  defp messages, do: [%{role: "user", content: "what's the answer?"}]

  defp usage(prompt, completion),
    do: %{prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion}

  defp tool_call_body(answer, usage) do
    args = if answer, do: %{answer: answer}, else: %{}

    %{
      id: "chatcmpl-test",
      object: "chat.completion",
      created: 1,
      model: "model-1",
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
      ]
    }
    |> maybe_usage(usage)
    |> Jason.encode!()
  end

  defp text_body(content, usage) do
    %{
      id: "chatcmpl-test",
      object: "chat.completion",
      created: 1,
      model: "model-1",
      choices: [
        %{index: 0, message: %{role: "assistant", content: content}, finish_reason: "stop"}
      ]
    }
    |> maybe_usage(usage)
    |> Jason.encode!()
  end

  defp maybe_usage(body, nil), do: body
  defp maybe_usage(body, usage), do: Map.put(body, :usage, usage)

  # Replies with each response in turn and forwards every request body to
  # the test process.
  defp expect_chat(bypass, responses) do
    test_pid = self()
    counter = :atomics.new(1, [])

    Bypass.expect(bypass, "POST", "/v1/chat/completions", fn conn ->
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      send(test_pid, {:request_body, body})
      idx = :atomics.add_get(counter, 1, 1) - 1
      {status, reply} = Enum.at(responses, idx)

      conn
      |> Plug.Conn.put_resp_content_type("application/json")
      |> Plug.Conn.resp(status, reply)
    end)
  end

  defp stop_metadata(base) do
    assert_receive {:telemetry, ^base, %{duration: duration}, metadata}
    assert is_integer(duration)
    metadata
  end

  describe "complete_structured/3" do
    test "stop metadata carries provider, model, role, usage, result and tags",
         %{bypass: bypass, provider: provider} do
      expect_chat(bypass, [{200, tool_call_body("42", usage(11, 7))}])

      assert {:ok, %TestSchema{answer: "42"}} =
               Client.complete_structured(TestSchema, messages(),
                 provider: provider,
                 telemetry_tags: %{feature: :quiz}
               )

      assert %{
               schema: TestSchema,
               provider: "acme",
               model: "model-1",
               usage: %{input_tokens: 11, output_tokens: 7},
               result: :ok,
               tags: %{feature: :quiz}
             } = stop_metadata([:defacto_ai, :complete_structured, :stop])
    end

    test "start metadata carries the tags", %{bypass: bypass, provider: provider} do
      expect_chat(bypass, [{200, tool_call_body("42", usage(1, 1))}])

      Client.complete_structured(TestSchema, messages(),
        provider: provider,
        telemetry_tags: %{feature: :quiz}
      )

      assert_receive {:telemetry, [:defacto_ai, :complete_structured, :start], _,
                      %{schema: TestSchema, tags: %{feature: :quiz}}}
    end

    test "the role comes from the role option and the resolver", %{
      bypass: bypass,
      provider: provider
    } do
      expect_chat(bypass, [{200, tool_call_body("42", usage(1, 1))}])
      previous = Application.get_env(:defacto_ai, :provider_resolver)
      Application.put_env(:defacto_ai, :provider_resolver, fn :summary -> provider end)

      on_exit(fn ->
        if previous,
          do: Application.put_env(:defacto_ai, :provider_resolver, previous),
          else: Application.delete_env(:defacto_ai, :provider_resolver)
      end)

      assert {:ok, _} = Client.complete_structured(TestSchema, messages(), role: :summary)

      assert %{role: :summary, provider: "acme", model: "model-1"} =
               stop_metadata([:defacto_ai, :complete_structured, :stop])
    end

    test "usage is summed over validation retries", %{bypass: bypass, provider: provider} do
      expect_chat(bypass, [
        {200, tool_call_body(nil, usage(10, 3))},
        {200, tool_call_body("fixed", usage(14, 4))}
      ])

      assert {:ok, %TestSchema{answer: "fixed"}} =
               Client.complete_structured(TestSchema, messages(), provider: provider)

      assert %{usage: %{input_tokens: 24, output_tokens: 7}, result: :ok} =
               stop_metadata([:defacto_ai, :complete_structured, :stop])
    end

    test "usage of rounds that ended in a validation error still counts",
         %{bypass: bypass, provider: provider} do
      expect_chat(bypass, [
        {200, tool_call_body(nil, usage(5, 1))},
        {200, tool_call_body(nil, usage(5, 1))},
        {200, tool_call_body(nil, usage(5, 1))}
      ])

      assert {:error, _} =
               Client.complete_structured(TestSchema, messages(),
                 provider: provider,
                 max_validation_retries: 2
               )

      # three tool-call rounds (initial + 2 retries), all billed
      assert %{usage: %{input_tokens: 15, output_tokens: 3}, result: :error} =
               stop_metadata([:defacto_ai, :complete_structured, :stop])
    end

    test "usage is nil when the provider reports none", %{bypass: bypass, provider: provider} do
      expect_chat(bypass, [{200, tool_call_body("42", nil)}])

      assert {:ok, _} = Client.complete_structured(TestSchema, messages(), provider: provider)

      assert %{usage: %{input_tokens: nil, output_tokens: nil}, result: :ok} =
               stop_metadata([:defacto_ai, :complete_structured, :stop])
    end

    test "tags default to an empty map", %{bypass: bypass, provider: provider} do
      expect_chat(bypass, [{200, tool_call_body("42", nil)}])
      Client.complete_structured(TestSchema, messages(), provider: provider)

      assert %{tags: tags} = stop_metadata([:defacto_ai, :complete_structured, :stop])
      assert tags == %{}
    end

    test "tags are not sent to the provider", %{bypass: bypass, provider: provider} do
      expect_chat(bypass, [{200, tool_call_body("42", nil)}])

      Client.complete_structured(TestSchema, messages(),
        provider: provider,
        telemetry_tags: %{feature: :secret_feature_name}
      )

      assert_receive {:request_body, body}
      refute body =~ "telemetry_tags"
      refute body =~ "secret_feature_name"
    end

    test "an error result is reported with the usage that was billed",
         %{bypass: bypass, provider: provider} do
      error = Jason.encode!(%{error: %{message: "server exploded", type: "server_error"}})
      expect_chat(bypass, [{500, error}])

      assert {:error, _} = Client.complete_structured(TestSchema, messages(), provider: provider)

      assert %{
               result: :error,
               provider: "acme",
               model: "model-1",
               usage: %{input_tokens: nil, output_tokens: nil}
             } = stop_metadata([:defacto_ai, :complete_structured, :stop])
    end

    test "an unresolvable provider is an error with nil provider and model" do
      previous = Application.get_env(:defacto_ai, :provider_resolver)
      Application.delete_env(:defacto_ai, :provider_resolver)

      on_exit(fn ->
        if previous, do: Application.put_env(:defacto_ai, :provider_resolver, previous)
      end)

      assert {:error, :no_provider_resolver} =
               Client.complete_structured(TestSchema, messages(), [])

      assert %{result: :error, provider: nil, model: nil, role: :llm} =
               stop_metadata([:defacto_ai, :complete_structured, :stop])
    end
  end

  describe "complete_chat/2" do
    test "stop metadata carries provider, model, role, usage, result and tags",
         %{bypass: bypass, provider: provider} do
      expect_chat(bypass, [{200, text_body("hello", usage(9, 2))}])

      assert {:ok, "hello"} =
               Client.complete_chat(messages(),
                 provider: provider,
                 telemetry_tags: %{feature: :chatty}
               )

      assert %{
               provider: "acme",
               model: "model-1",
               usage: %{input_tokens: 9, output_tokens: 2},
               result: :ok,
               tags: %{feature: :chatty}
             } = stop_metadata([:defacto_ai, :complete_chat, :stop])
    end

    test "usage is nil when the provider reports none", %{bypass: bypass, provider: provider} do
      expect_chat(bypass, [{200, text_body("hello", nil)}])

      assert {:ok, "hello"} = Client.complete_chat(messages(), provider: provider)

      assert %{usage: %{input_tokens: nil, output_tokens: nil}} =
               stop_metadata([:defacto_ai, :complete_chat, :stop])
    end

    test "tags are not sent to the provider", %{bypass: bypass, provider: provider} do
      expect_chat(bypass, [{200, text_body("hello", nil)}])

      Client.complete_chat(messages(),
        provider: provider,
        telemetry_tags: %{feature: :secret_feature_name}
      )

      assert_receive {:request_body, body}
      refute body =~ "telemetry_tags"
      refute body =~ "secret_feature_name"
    end

    test "an error result", %{bypass: bypass, provider: provider} do
      expect_chat(bypass, [{500, Jason.encode!(%{error: %{message: "boom"}})}])

      assert {:error, _} = Client.complete_chat(messages(), provider: provider)

      assert %{result: :error, usage: %{input_tokens: nil, output_tokens: nil}} =
               stop_metadata([:defacto_ai, :complete_chat, :stop])
    end
  end

  describe "embed/2 and embed_batch/2" do
    setup do
      Req.Test.set_req_test_to_private()
      :ok
    end

    defp stub_embeddings(usage) do
      test_pid = self()

      Req.Test.stub(__MODULE__, fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        send(test_pid, {:request_body, body})

        reply =
          %{data: [%{index: 0, embedding: [0.1, 0.2]}], model: "model-1"}
          |> then(fn reply -> if usage, do: Map.put(reply, :usage, usage), else: reply end)

        Req.Test.json(conn, reply)
      end)
    end

    test "embed stop metadata carries provider, model, role, usage, result and tags",
         %{provider: provider} do
      stub_embeddings(%{prompt_tokens: 8, total_tokens: 8})

      assert {:ok, [0.1, 0.2]} =
               Embeddings.embed("hello there",
                 provider: provider,
                 plug: {Req.Test, __MODULE__},
                 telemetry_tags: %{feature: :search}
               )

      assert %{
               provider: "acme",
               model: "model-1",
               role: :embedding,
               usage: %{input_tokens: 8, output_tokens: nil},
               result: :ok,
               tags: %{feature: :search}
             } = stop_metadata([:defacto_ai, :embed, :stop])
    end

    test "embed usage is nil when the provider reports none", %{provider: provider} do
      stub_embeddings(nil)

      assert {:ok, _} =
               Embeddings.embed("hello there", provider: provider, plug: {Req.Test, __MODULE__})

      assert %{usage: %{input_tokens: nil, output_tokens: nil}, tags: tags} =
               stop_metadata([:defacto_ai, :embed, :stop])

      assert tags == %{}
    end

    test "embed tags are not sent to the provider", %{provider: provider} do
      stub_embeddings(nil)

      Embeddings.embed("hello there",
        provider: provider,
        plug: {Req.Test, __MODULE__},
        telemetry_tags: %{feature: :secret_feature_name}
      )

      assert_receive {:request_body, body}
      refute body =~ "telemetry_tags"
      refute body =~ "secret_feature_name"
    end

    test "embed error result", %{provider: provider} do
      Req.Test.stub(__MODULE__, fn conn -> Plug.Conn.send_resp(conn, 500, "nope") end)

      assert {:error, {:api_error, 500, _}} =
               Embeddings.embed("hello there", provider: provider, plug: {Req.Test, __MODULE__})

      assert %{result: :error, usage: %{input_tokens: nil, output_tokens: nil}} =
               stop_metadata([:defacto_ai, :embed, :stop])
    end

    test "embed_batch sums usage over the texts and keeps count", %{provider: provider} do
      stub_embeddings(%{prompt_tokens: 5, total_tokens: 5})

      assert {:ok, [_, _, _]} =
               Embeddings.embed_batch(["one", "two", "three"],
                 provider: provider,
                 plug: {Req.Test, __MODULE__},
                 telemetry_tags: %{feature: :indexing}
               )

      assert %{
               count: 3,
               provider: "acme",
               model: "model-1",
               usage: %{input_tokens: 15, output_tokens: nil},
               result: :ok,
               tags: %{feature: :indexing}
             } = stop_metadata([:defacto_ai, :embed_batch, :stop])
    end

    test "embed_batch error result keeps the usage of texts that succeeded",
         %{provider: provider} do
      counter = :atomics.new(1, [])

      Req.Test.stub(__MODULE__, fn conn ->
        if :atomics.add_get(counter, 1, 1) == 1 do
          Req.Test.json(conn, %{
            data: [%{index: 0, embedding: [0.1]}],
            usage: %{prompt_tokens: 4, total_tokens: 4}
          })
        else
          Plug.Conn.send_resp(conn, 500, "nope")
        end
      end)

      assert {:error, _} =
               Embeddings.embed_batch(["one", "two"],
                 provider: provider,
                 plug: {Req.Test, __MODULE__}
               )

      assert %{result: :error, count: 2, usage: %{input_tokens: 4, output_tokens: nil}} =
               stop_metadata([:defacto_ai, :embed_batch, :stop])
    end
  end
end
