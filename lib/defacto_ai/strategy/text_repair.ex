defmodule DefactoAI.Strategy.TextRepair do
  @moduledoc """
  Last-resort strategy: ask the model for JSON in the prompt and trust
  `DefactoAI.JSON` to extract / repair whatever comes back.

  No structured-output flags are sent. Works against any chat-completion
  endpoint, including small or self-hosted models that don't speak tool
  calling or JSON mode.
  """

  @behaviour DefactoAI.Strategy

  alias DefactoAI.LangChainAdapter

  @impl true
  def prepare(provider, schema_module, messages, _opts) do
    chat_model = LangChainAdapter.build_chat_model(provider)

    chain = LangChainAdapter.build_chain(chat_model, prepend_system_prompt(messages, schema_module))

    {:ok, chain}
  end

  @impl true
  def decode_payload(%LangChain.Chains.LLMChain{last_message: nil}), do: {:error, :no_message}

  def decode_payload(%LangChain.Chains.LLMChain{last_message: %{content: nil}}),
    do: {:error, :no_content}

  def decode_payload(%LangChain.Chains.LLMChain{last_message: %{content: content}})
      when is_binary(content),
      do: {:ok, content}

  def decode_payload(%LangChain.Chains.LLMChain{last_message: %{content: parts}})
      when is_list(parts) do
    text =
      parts
      |> Enum.map(fn
        %LangChain.Message.ContentPart{type: :text, content: c} when is_binary(c) -> c
        _ -> ""
      end)
      |> Enum.join()

    if text == "", do: {:error, :no_content}, else: {:ok, text}
  end

  defp prepend_system_prompt(messages, schema_module) do
    Code.ensure_loaded(schema_module)

    schema_text =
      if function_exported?(schema_module, :parameters_schema, 0) do
        Jason.encode!(schema_module.parameters_schema(), pretty: true)
      else
        "(no schema available)"
      end

    system = %{
      role: :system,
      content: """
      You must respond with a single valid JSON value and nothing else.
      The value must conform to this JSON schema:

      #{schema_text}

      Strict requirements:
      - No prose, no greeting, no commentary.
      - No markdown, no code fences, no backticks.
      - Use straight ASCII double quotes (not curly quotes).
      - No trailing commas.
      """
    }

    [system | messages]
  end
end
