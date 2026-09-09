defmodule DefactoAI.StreamRunnerTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias DefactoAI.StreamRunner
  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ToolCall

  @error_body ~s({"error":{"code":422,"message":"failed to fetch image; check the url provided is valid","type":"unprocessable_entity"}})

  defp chain do
    llm =
      ChatOpenAI.new!(%{
        endpoint: "https://api.example.com/v1/chat/completions",
        api_key: "sk-test",
        model: "gpt-test",
        stream: true
      })

    LLMChain.new!(%{llm: llm})
    |> LLMChain.add_message(Message.new_user!("describe this image"))
  end

  defp sse_body(json_chunks) do
    json_chunks
    |> Enum.map(&"data: #{&1}\n\n")
    |> Enum.join()
    |> Kernel.<>("data: [DONE]\n\n")
  end

  defp stub_sse(json_chunks) do
    Req.Test.stub(__MODULE__, fn conn ->
      conn
      |> Plug.Conn.put_resp_content_type("text/event-stream")
      |> Plug.Conn.send_resp(200, sse_body(json_chunks))
    end)
  end

  defp delta(delta_map), do: Jason.encode!(%{choices: [%{index: 0, delta: delta_map}]})

  defp opening_chunk(index, id, name) do
    delta(%{
      tool_calls: [
        %{index: index, id: id, type: "function", function: %{name: name, arguments: ""}}
      ]
    })
  end

  defp run_tool_calls(json_chunks) do
    stub_sse(json_chunks)

    assert {:ok, %LLMChain{last_message: %Message{tool_calls: tool_calls}}} =
             StreamRunner.run(chain(), plug: {Req.Test, __MODULE__})

    tool_calls
  end

  describe "run/2 — tool-call deltas" do
    test "assembles the standard OpenAI shape with an index on every chunk" do
      tool_calls =
        run_tool_calls([
          opening_chunk(0, "call_1", "respond"),
          delta(%{tool_calls: [%{index: 0, function: %{arguments: ~s({"answer":)}}]}),
          delta(%{tool_calls: [%{index: 0, function: %{arguments: ~s("42"})}}]})
        ])

      assert [%ToolCall{call_id: "call_1", name: "respond", arguments: %{"answer" => "42"}}] =
               tool_calls
    end

    test "attaches index-less argument deltas to the most recently opened call" do
      # Anthropic-backed gateways open the call with index/id/name and then
      # stream argument deltas without any index.
      tool_calls =
        run_tool_calls([
          opening_chunk(0, "call_1", "respond"),
          delta(%{tool_calls: [%{function: %{arguments: ~s({"answer":)}}]}),
          delta(%{tool_calls: [%{function: %{arguments: ~s("42"})}}]})
        ])

      assert [%ToolCall{call_id: "call_1", name: "respond", arguments: %{"answer" => "42"}}] =
               tool_calls
    end

    test "attaches argument deltas whose index never opened a call to the current call" do
      tool_calls =
        run_tool_calls([
          opening_chunk(0, "call_1", "respond"),
          delta(%{tool_calls: [%{index: 1, function: %{arguments: ~s({"answer":)}}]}),
          delta(%{tool_calls: [%{index: 1, function: %{arguments: ~s("42"})}}]})
        ])

      assert [%ToolCall{call_id: "call_1", name: "respond", arguments: %{"answer" => "42"}}] =
               tool_calls
    end

    test "treats a blank function name on argument deltas as absent (Heroku Inference shape)" do
      # Ground truth from a live probe: frames are `event: message` +
      # `data: {...}`, the opening chunk carries index/id/name, and every
      # argument delta repeats `"name": ""`.
      frames =
        [
          %{
            delta: %{
              tool_calls: [
                %{
                  index: 0,
                  id: "tooluse_1",
                  type: "function",
                  function: %{name: "respond", arguments: ""}
                }
              ]
            },
            index: 0
          },
          %{
            delta: %{tool_calls: [%{index: 0, function: %{name: "", arguments: ~s({"ans)}}]},
            index: 0
          },
          %{
            delta: %{tool_calls: [%{index: 0, function: %{name: "", arguments: ~s(wer":"42"})}}]},
            index: 0
          }
        ]
        |> Enum.map(&%{choices: [&1]})
        |> Enum.map(&Jason.encode!/1)
        |> Enum.map_join("", &"event: message\ndata: #{&1}\n\n")
        |> Kernel.<>("event: message\ndata: [DONE]\n\n")

      Req.Test.stub(__MODULE__, fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_resp(200, frames)
      end)

      assert {:ok, %LLMChain{last_message: %Message{tool_calls: tool_calls}}} =
               StreamRunner.run(chain(), plug: {Req.Test, __MODULE__})

      assert [%ToolCall{call_id: "tooluse_1", name: "respond", arguments: %{"answer" => "42"}}] =
               tool_calls
    end

    test "merges deltas onto the started call when a text block shifts their index" do
      # Anthropic content-block indexes: text block 0, tool_use block 1. The
      # gateway opens the call at index 1 but streams argument deltas at 0.
      tool_calls =
        run_tool_calls([
          delta(%{content: "Here you go: "}),
          opening_chunk(1, "tooluse_1", "respond"),
          delta(%{tool_calls: [%{index: 0, function: %{name: "", arguments: ~s({"answer":)}}]}),
          delta(%{tool_calls: [%{index: 0, function: %{name: "", arguments: ~s("42"})}}]})
        ])

      assert [
               %ToolCall{
                 index: 1,
                 call_id: "tooluse_1",
                 name: "respond",
                 arguments: %{"answer" => "42"}
               }
             ] =
               tool_calls
    end

    test "folds argument deltas that arrived before the call opened into it" do
      tool_calls =
        run_tool_calls([
          delta(%{tool_calls: [%{index: 0, function: %{name: "", arguments: ~s({"answer":)}}]}),
          opening_chunk(1, "tooluse_1", "respond"),
          delta(%{tool_calls: [%{index: 0, function: %{name: "", arguments: ~s("42"})}}]})
        ])

      assert [%ToolCall{call_id: "tooluse_1", name: "respond", arguments: %{"answer" => "42"}}] =
               tool_calls
    end

    test "keeps properly indexed parallel tool calls apart" do
      tool_calls =
        run_tool_calls([
          opening_chunk(0, "call_a", "first"),
          opening_chunk(1, "call_b", "second"),
          delta(%{tool_calls: [%{index: 0, function: %{arguments: ~s({"n":1})}}]}),
          delta(%{tool_calls: [%{index: 1, function: %{arguments: ~s({"n":2})}}]})
        ])

      assert [
               %ToolCall{index: 0, call_id: "call_a", name: "first", arguments: %{"n" => 1}},
               %ToolCall{index: 1, call_id: "call_b", name: "second", arguments: %{"n" => 2}}
             ] = tool_calls
    end

    test "accepts a complete (non-delta) message with finished tool calls" do
      complete =
        Jason.encode!(%{
          choices: [
            %{
              index: 0,
              message: %{
                role: "assistant",
                content: nil,
                tool_calls: [
                  %{
                    id: "call_9",
                    type: "function",
                    function: %{name: "respond", arguments: ~s({"answer":"x"})}
                  }
                ]
              }
            }
          ]
        })

      assert [%ToolCall{call_id: "call_9", name: "respond", arguments: %{"answer" => "x"}}] =
               run_tool_calls([complete])
    end

    test "accepts the legacy function_call shape as tool call 0" do
      tool_calls =
        run_tool_calls([
          delta(%{function_call: %{name: "respond", arguments: ""}}),
          delta(%{function_call: %{arguments: ~s({"answer":"y"})}})
        ])

      assert [%ToolCall{index: 0, name: "respond", arguments: %{"answer" => "y"}}] = tool_calls
    end

    test "logs a bounded picture of the stream when a tool call ends with empty arguments" do
      stub_sse([
        # Not a delta/message chunk: counted and sampled as unrecognised.
        ~s({"type":"ping"}),
        delta(%{content: "Here is the answer"}),
        opening_chunk(0, "call_1", "respond")
      ])

      log =
        capture_log([level: :info], fn ->
          assert {:ok, %LLMChain{last_message: %Message{tool_calls: [%ToolCall{}]}}} =
                   StreamRunner.run(chain(), plug: {Req.Test, __MODULE__})
        end)

      assert log =~ "DefactoAI.StreamRunner: streamed tool call finished with empty arguments"
      assert log =~ "content_length=18"
      assert log =~ "frames=3 unrecognised=1"
      assert log =~ "1 raw tool-call chunk(s) seen"
      assert log =~ ~s("id" => "call_1")
      assert log =~ ~s(content_sample: "Here is the answer")
      assert log =~ ~S(unrecognised_sample: "{\"type\":\"ping\"}")
    end

    test "stays quiet when the tool call carries arguments" do
      log =
        capture_log([level: :info], fn ->
          run_tool_calls([
            opening_chunk(0, "call_1", "respond"),
            delta(%{tool_calls: [%{index: 0, function: %{arguments: ~s({"answer":"42"})}}]})
          ])
        end)

      refute log =~ "empty arguments"
    end
  end

  describe "run/2" do
    test "assembles streamed content deltas into an assistant message" do
      Req.Test.stub(__MODULE__, fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_resp(
          200,
          sse_body([
            ~s({"choices":[{"delta":{"content":"Hello, "}}]}),
            ~s({"choices":[{"delta":{"content":"world!"}}]})
          ])
        )
      end)

      assert {:ok, %LLMChain{last_message: %Message{role: :assistant, content: "Hello, world!"}}} =
               StreamRunner.run(chain(), plug: {Req.Test, __MODULE__})
    end

    test "preserves the provider's error body on non-200 responses" do
      # Heroku Inference answers a bad image URL with a 422 and a JSON error
      # body. The SSE collector only reads `data:` lines, so without buffering
      # the raw body the caller would see {:api_error, 422, ""} and lose the
      # message it classifies on.
      Req.Test.stub(__MODULE__, fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.send_resp(422, @error_body)
      end)

      assert {:error, %LLMChain{}, {:api_error, 422, body}} =
               StreamRunner.run(chain(), plug: {Req.Test, __MODULE__})

      assert body == @error_body
      assert body =~ "failed to fetch image"
    end

    test "falls back to an empty body when the error response has no body" do
      Req.Test.stub(__MODULE__, fn conn -> Plug.Conn.send_resp(conn, 503, "") end)

      assert {:error, %LLMChain{}, {:api_error, 503, ""}} =
               StreamRunner.run(chain(), plug: {Req.Test, __MODULE__})
    end
  end
end
