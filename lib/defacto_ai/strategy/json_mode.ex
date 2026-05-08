defmodule DefactoAI.Strategy.JsonMode do
  @moduledoc """
  Structured-output strategy that enables OpenAI's
  `response_format: json_object` (LangChain's `json_response: true`).

  The model must include the word "json" somewhere in the prompt for
  this to work on OpenAI proper, so we prepend a system message
  instructing the model to respond as JSON matching the response shape.
  """

  @behaviour DefactoAI.Strategy

  alias DefactoAI.LangChainAdapter

  @impl true
  def prepare(provider, schema_module, messages, _opts) do
    chat_model = LangChainAdapter.build_chat_model(provider, json_response: true)

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
      Respond with a single JSON object and nothing else. The object must
      conform to this JSON schema:

      #{schema_text}

      Do not include prose, markdown, or code fences. JSON only.
      """
    }

    [system | messages]
  end
end
