defmodule DefactoAI.RepairLoopTest do
  @moduledoc """
  Drives `DefactoAI.RepairLoop.run/5` through the real streaming request path
  (Req `plug:` stub) so we can assert on the exact message history sent
  upstream and on what the loop logs / emits when validation fails.
  """

  # Toggles the global :chat_stream env, so it must not run alongside tests
  # that read it.
  use ExUnit.Case, async: false

  import ExUnit.CaptureLog

  alias DefactoAI.RepairLoop
  alias DefactoAI.Strategy
  alias DefactoAI.TestProvider
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult

  @placeholder "(structured response provided as a tool call)"

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

  setup do
    Application.put_env(:defacto_ai, :chat_stream, true)
    on_exit(fn -> Application.delete_env(:defacto_ai, :chat_stream) end)
    :ok
  end

  defp chain(history \\ []) do
    {:ok, chain} =
      Strategy.ToolCall.prepare(
        TestProvider.new(),
        TestSchema,
        [%{role: "user", content: "what's the answer?"}],
        []
      )

    Enum.reduce(history, chain, &LLMChain.add_message(&2, &1))
  end

  defp tool_call_turn(content) do
    %Message{
      role: :assistant,
      status: :complete,
      content: content,
      tool_calls: [
        %ToolCall{
          type: :function,
          status: :complete,
          index: 0,
          call_id: "call_1",
          name: "respond",
          arguments: %{}
        }
      ]
    }
  end

  defp run(chain, budget \\ 0) do
    RepairLoop.run(chain, TestSchema, &Strategy.ToolCall.decode_payload/1, budget,
      plug: {Req.Test, __MODULE__}
    )
  end

  # Records every request body the loop sends and answers each one with a
  # streamed `respond` tool call whose arguments are taken from `arguments`
  # in order (the last entry repeats).
  defp stub_upstream(arguments) do
    test_pid = self()
    counter = :atomics.new(1, [])

    Req.Test.stub(__MODULE__, fn conn ->
      {:ok, raw, conn} = Plug.Conn.read_body(conn)
      send(test_pid, {:request, Jason.decode!(raw)})

      idx = :atomics.add_get(counter, 1, 1) - 1
      args = Enum.at(arguments, idx) || List.last(arguments)

      conn
      |> Plug.Conn.put_resp_content_type("text/event-stream")
      |> Plug.Conn.send_resp(200, sse_body([tool_call_chunk(args)]))
    end)
  end

  defp tool_call_chunk(arguments_json) do
    Jason.encode!(%{
      choices: [
        %{
          index: 0,
          delta: %{
            tool_calls: [
              %{
                index: 0,
                id: "call_2",
                type: "function",
                function: %{name: "respond", arguments: arguments_json}
              }
            ]
          }
        }
      ]
    })
  end

  defp sse_body(json_chunks) do
    json_chunks
    |> Enum.map(&"data: #{&1}\n\n")
    |> Enum.join()
    |> Kernel.<>("data: [DONE]\n\n")
  end

  defp sent_assistant_message do
    assert_received {:request, %{"messages" => messages}}
    Enum.find(messages, &(&1["role"] == "assistant"))
  end

  describe "run/5 — re-sent assistant tool-call turns" do
    test "substitutes a placeholder for nil content when the turn carries tool calls" do
      stub_upstream([~s({"answer":"42"})])

      assert {:ok, %TestSchema{answer: "42"}} = run(chain([tool_call_turn(nil)]))

      assistant = sent_assistant_message()
      assert assistant["content"] == @placeholder
      assert [%{"function" => %{"name" => "respond"}}] = assistant["tool_calls"]
    end

    test "substitutes a placeholder for empty-string content when the turn carries tool calls" do
      stub_upstream([~s({"answer":"42"})])

      assert {:ok, %TestSchema{}} = run(chain([tool_call_turn("")]))

      assert sent_assistant_message()["content"] == @placeholder
    end

    test "substitutes a placeholder for empty-list content when the turn carries tool calls" do
      stub_upstream([~s({"answer":"42"})])

      assert {:ok, %TestSchema{}} = run(chain([tool_call_turn([])]))

      assert sent_assistant_message()["content"] == @placeholder
    end

    test "leaves assistant messages without tool calls untouched" do
      stub_upstream([~s({"answer":"42"})])

      plain = %Message{role: :assistant, status: :complete, content: "", tool_calls: []}

      assert {:ok, %TestSchema{}} = run(chain([plain]))

      assert_received {:request, %{"messages" => messages}}
      assistant = Enum.find(messages, &(&1["role"] == "assistant"))
      assert assistant["content"] == ""
      refute Enum.any?(messages, &(&1["content"] == @placeholder))
    end
  end

  # Wire content may be a plain string or a list of text parts.
  defp text_of(content) when is_binary(content), do: content
  defp text_of(parts) when is_list(parts), do: Enum.map_join(parts, "", &part_text/1)

  defp part_text(%{"text" => text}), do: text
  defp part_text(%LangChain.Message.ContentPart{content: text}) when is_binary(text), do: text
  defp part_text(_), do: ""

  describe "run/5 — corrective turn after a rejected tool call" do
    test "answers the tool call with a tool result before the corrective user message" do
      stub_upstream([~s({"wrong":"x"}), ~s({"answer":"42"})])

      assert {:ok, %TestSchema{answer: "42"}} = run(chain(), 1)

      # First request is the initial turn; the second carries the retry history.
      assert_received {:request, %{"messages" => _initial}}
      assert_received {:request, %{"messages" => messages}}

      [assistant, tool, user] = Enum.take(messages, -3)

      assert assistant["role"] == "assistant"
      assert [%{"id" => "call_2", "function" => %{"name" => "respond"}}] = assistant["tool_calls"]

      assert tool["role"] == "tool"
      assert tool["tool_call_id"] == "call_2"
      assert text_of(tool["content"]) =~ "answer: can't be blank"

      assert user["role"] == "user"
      assert text_of(user["content"]) =~ "could not be used"
    end

    test "builds one is_error tool result per tool call, then the user message" do
      changeset = TestSchema.changeset(%TestSchema{}, %{})

      assert [
               %Message{role: :tool, tool_results: [result]},
               %Message{role: :user}
             ] = RepairLoop.corrective_messages(chain([tool_call_turn(nil)]), changeset, %{})

      assert %ToolResult{tool_call_id: "call_1", name: "respond", is_error: true} = result
      assert text_of(result.content) =~ "answer: can't be blank"
    end

    test "sends only the corrective user message when the last answer was not a tool call" do
      changeset = TestSchema.changeset(%TestSchema{}, %{})
      plain = %Message{role: :assistant, status: :complete, content: "not json", tool_calls: []}

      assert [%Message{role: :user}] =
               RepairLoop.corrective_messages(chain([plain]), changeset, "not json")
    end
  end

  describe "run/5 — empty payloads" do
    test "treats an empty decoded payload as a decode failure, not a validation failure" do
      stub_upstream([~s({"answer":"42"})])
      decode_empty = fn _chain -> {:ok, %{}} end

      assert {:error, {:decode_failed, :empty_payload}} =
               RepairLoop.run(chain(), TestSchema, decode_empty, 2, plug: {Req.Test, __MODULE__})

      # Exactly one upstream call: no repair budget was spent.
      assert_received {:request, _}
      refute_received {:request, _}
    end
  end

  describe "run/5 — validation failure visibility" do
    test "logs the failure at :info with strategy, schema, errors and payload keys" do
      # First answer has the wrong key (so `answer` is blank), second is valid.
      stub_upstream([~s({"wrong":"x"}), ~s({"answer":"42"})])

      log =
        capture_log([level: :info], fn ->
          assert {:ok, %TestSchema{answer: "42"}} = run(chain(), 1)
        end)

      assert log =~ "[info]"
      assert log =~ "DefactoAI: validation failed, retrying with corrective message"
      assert log =~ "(0 attempts left)"
      assert log =~ "strategy=DefactoAI.Strategy.ToolCall"
      assert log =~ "schema=DefactoAI.RepairLoopTest.TestSchema"
      assert log =~ "answer: can't be blank"
      assert log =~ "payload_keys=[wrong]"
    end

    test "attaches the bounded reason to the :retry telemetry event" do
      stub_upstream([~s({"wrong":"x"}), ~s({"answer":"42"})])

      handler = "repair-loop-test-#{inspect(make_ref())}"
      test_pid = self()

      :telemetry.attach(
        handler,
        [:defacto_ai, :repair_loop, :retry],
        fn event, measurements, metadata, _ ->
          send(test_pid, {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      try do
        capture_log(fn -> assert {:ok, %TestSchema{}} = run(chain(), 1) end)

        assert_received {:telemetry, [:defacto_ai, :repair_loop, :retry], %{remaining: 0},
                         %{schema: TestSchema, reason: reason}}

        assert reason =~ "answer: can't be blank"
        assert reason =~ "payload_keys=[wrong]"
        assert String.length(reason) <= 500
      after
        :telemetry.detach(handler)
      end
    end
  end
end
