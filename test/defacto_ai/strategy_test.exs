defmodule DefactoAI.StrategyTest do
  use ExUnit.Case, async: true

  alias DefactoAI.Strategy
  alias DefactoAI.TestProvider
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Chains.LLMChain
  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.Message.ToolCall

  defmodule TestSchema do
    use Ecto.Schema
    import Ecto.Changeset

    @primary_key false
    embedded_schema do
      field(:answer, :string)
    end

    def changeset(struct, attrs, _opts \\ []) do
      cast(struct, attrs, [:answer])
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

  defp messages, do: [%{role: "user", content: "what's the meaning of life?"}]

  describe "default_order/0" do
    test "lists tool-call first, then json-mode, then text-repair" do
      assert Strategy.default_order() == [
               Strategy.ToolCall,
               Strategy.JsonMode,
               Strategy.TextRepair
             ]
    end
  end

  describe "ToolCall.prepare/4" do
    test "registers a single 'respond' tool with the schema's parameters_schema" do
      assert {:ok, %LLMChain{} = chain} =
               Strategy.ToolCall.prepare(TestProvider.new(), TestSchema, messages(), [])

      assert [%Function{name: "respond", parameters_schema: schema}] = chain.tools
      assert schema == TestSchema.parameters_schema()
    end

    test "strips top-level description from parameters_schema (some providers reject it)" do
      defmodule SchemaWithTopLevelDescription do
        def parameters_schema do
          %{
            type: "object",
            description: "A tool that does X.",
            properties: %{value: %{type: "string", description: "kept"}},
            required: ["value"]
          }
        end

        def changeset(struct, attrs, _opts \\ []), do: Ecto.Changeset.cast(struct, attrs, [])
      end

      {:ok, chain} =
        Strategy.ToolCall.prepare(TestProvider.new(), SchemaWithTopLevelDescription, messages(), [])

      [%Function{description: tool_desc, parameters_schema: params}] = chain.tools

      # Top-level :description is removed from parameters but preserved on
      # the function itself so the prompt-engineering benefit isn't lost.
      refute Map.has_key?(params, :description)
      refute Map.has_key?(params, "description")
      assert tool_desc == "A tool that does X."
      # Per-property descriptions are kept.
      assert params.properties.value.description == "kept"
    end

    test "forces tool_choice to the 'respond' function" do
      {:ok, chain} = Strategy.ToolCall.prepare(TestProvider.new(), TestSchema, messages(), [])

      assert %ChatOpenAI{
               tool_choice: %{"type" => "function", "function" => %{"name" => "respond"}}
             } = chain.llm
    end

    test "errors when the schema doesn't expose parameters_schema/0" do
      defmodule NoSchema do
        defstruct []
      end

      assert {:error, {:invalid_schema, {NoSchema, :parameters_schema, 0}}} =
               Strategy.ToolCall.prepare(TestProvider.new(), NoSchema, messages(), [])
    end
  end

  describe "ToolCall.decode_payload/1" do
    test "returns the parsed arguments map from the assistant's tool call" do
      args = %{"answer" => "42"}

      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{
          role: :assistant,
          tool_calls: [%ToolCall{name: "respond", arguments: args, status: :complete}]
        }
      }

      assert {:ok, ^args} = Strategy.ToolCall.decode_payload(chain)
    end

    test "errors when the model didn't make any tool call" do
      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{role: :assistant, tool_calls: []}
      }

      assert {:error, :no_tool_call} = Strategy.ToolCall.decode_payload(chain)
    end

    test "errors when the model called a different tool" do
      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{
          role: :assistant,
          tool_calls: [%ToolCall{name: "other", arguments: %{}, status: :complete}]
        }
      }

      assert {:error, :wrong_tool_call} = Strategy.ToolCall.decode_payload(chain)
    end

    test "errors when there is no last message" do
      chain = %LLMChain{llm: %ChatOpenAI{}, last_message: nil}
      assert {:error, :no_message} = Strategy.ToolCall.decode_payload(chain)
    end

    test "hands unparsed argument text to the JSON pipeline" do
      truncated = ~s({"answer": "4)

      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{
          role: :assistant,
          tool_calls: [%ToolCall{name: "respond", arguments: truncated, status: :complete}]
        }
      }

      assert {:ok, ^truncated} = Strategy.ToolCall.decode_payload(chain)
    end

    test "falls back to the message content when the respond call has no arguments" do
      content = "```json\n{\"answer\": \"42\"}\n```"

      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{
          role: :assistant,
          content: content,
          tool_calls: [%ToolCall{name: "respond", arguments: %{}, status: :complete}]
        }
      }

      assert {:ok, ^content} = Strategy.ToolCall.decode_payload(chain)
    end

    test "falls back to text content parts when the respond call has no arguments" do
      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{
          role: :assistant,
          content: [LangChain.Message.ContentPart.text!(~s({"answer": "42"}))],
          tool_calls: [%ToolCall{name: "respond", arguments: nil, status: :complete}]
        }
      }

      assert {:ok, ~s({"answer": "42"})} = Strategy.ToolCall.decode_payload(chain)
    end

    test "errors :empty_tool_call when neither arguments nor content carry anything" do
      for {arguments, content} <- [{%{}, nil}, {nil, ""}, {"", "  "}, {%{}, []}] do
        chain = %LLMChain{
          llm: %ChatOpenAI{},
          last_message: %Message{
            role: :assistant,
            content: content,
            tool_calls: [%ToolCall{name: "respond", arguments: arguments, status: :complete}]
          }
        }

        assert {:error, :empty_tool_call} = Strategy.ToolCall.decode_payload(chain)
      end
    end
  end

  describe "ToolCall.decode_payload/1 with several tool calls" do
    test "prefers the respond call that carries arguments" do
      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{
          role: :assistant,
          tool_calls: [
            %ToolCall{name: "respond", call_id: "call_1", arguments: %{}},
            %ToolCall{name: "respond", call_id: "call_2", arguments: %{"answer" => "42"}}
          ]
        }
      }

      assert {:ok, %{"answer" => "42"}} = Strategy.ToolCall.decode_payload(chain)
    end

    test "prefers a respond call over an unrelated tool call listed first" do
      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{
          role: :assistant,
          tool_calls: [
            %ToolCall{name: "other", call_id: "call_1", arguments: %{"x" => 1}},
            %ToolCall{name: "respond", call_id: "call_2", arguments: %{"answer" => "42"}}
          ]
        }
      }

      assert {:ok, %{"answer" => "42"}} = Strategy.ToolCall.decode_payload(chain)
    end
  end

  describe "JsonMode.prepare/4" do
    test "enables json_response on the chat model" do
      assert {:ok, %LLMChain{llm: %ChatOpenAI{json_response: true}}} =
               Strategy.JsonMode.prepare(TestProvider.new(), TestSchema, messages(), [])
    end

    test "prepends a JSON-only system message" do
      {:ok, chain} = Strategy.JsonMode.prepare(TestProvider.new(), TestSchema, messages(), [])

      assert [%Message{role: :system, content: system_content}, %Message{role: :user} | _] =
               chain.messages

      system_text = system_text_to_binary(system_content)
      assert system_text =~ "JSON"
      assert system_text =~ "answer"
    end
  end

  describe "JsonMode.decode_payload/1" do
    test "returns the assistant's text content" do
      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{role: :assistant, content: ~s({"answer": "42"})}
      }

      assert {:ok, ~s({"answer": "42"})} = Strategy.JsonMode.decode_payload(chain)
    end

    test "concatenates multi-part text content" do
      parts = [
        %LangChain.Message.ContentPart{type: :text, content: ~s({"answer":)},
        %LangChain.Message.ContentPart{type: :text, content: ~s( "42"})}
      ]

      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{role: :assistant, content: parts}
      }

      assert {:ok, ~s({"answer": "42"})} = Strategy.JsonMode.decode_payload(chain)
    end

    test "errors when the assistant's content is missing" do
      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{role: :assistant, content: nil}
      }

      assert {:error, :no_content} = Strategy.JsonMode.decode_payload(chain)
    end
  end

  describe "TextRepair.prepare/4" do
    test "leaves json_response off and adds no tools" do
      assert {:ok, %LLMChain{llm: %ChatOpenAI{json_response: false}, tools: []}} =
               Strategy.TextRepair.prepare(TestProvider.new(), TestSchema, messages(), [])
    end

    test "prepends a strict JSON-only system message" do
      {:ok, chain} = Strategy.TextRepair.prepare(TestProvider.new(), TestSchema, messages(), [])
      [%Message{role: :system, content: system_content} | _] = chain.messages
      system_text = system_text_to_binary(system_content)
      assert system_text =~ "JSON"
      assert system_text =~ "no code fences"
      assert system_text =~ "answer"
    end
  end

  describe "TextRepair.decode_payload/1" do
    test "returns the assistant's text content unchanged" do
      raw = "Sure! Here you go: {\"answer\": \"42\"}"

      chain = %LLMChain{
        llm: %ChatOpenAI{},
        last_message: %Message{role: :assistant, content: raw}
      }

      assert {:ok, ^raw} = Strategy.TextRepair.decode_payload(chain)
    end
  end

  defp system_text_to_binary(text) when is_binary(text), do: text

  defp system_text_to_binary(parts) when is_list(parts) do
    parts
    |> Enum.map(fn
      %LangChain.Message.ContentPart{type: :text, content: c} -> c
      _ -> ""
    end)
    |> Enum.join()
  end
end
