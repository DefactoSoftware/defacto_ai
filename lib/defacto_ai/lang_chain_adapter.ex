defmodule DefactoAI.LangChainAdapter do
  @moduledoc """
  Maps a `DefactoAI.Provider` value onto a LangChain
  `LangChain.ChatModels.ChatOpenAI` chat model and assembles an
  `LLMChain` from a list of messages.

  Strategy modules (`DefactoAI.Strategy.*`) call `build_chat_model/2`
  with strategy-specific overrides like `json_response: true`,
  `tools: [...]`, or `tool_choice: ...`, and call `build_chain/2` to wrap
  the chat model with the user's messages.

  The chat model accepts an arbitrary `endpoint` URL — that's documented
  for Azure / self-hosted OpenAI-compatible providers — so we can point
  at any provider's `base_url <> api_path`.
  """

  alias DefactoAI.Provider
  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message

  @default_receive_timeout 180_000

  @type message_input ::
          Message.t() | %{required(:role) => atom() | binary(), required(:content) => binary()}

  @doc """
  Build a `ChatOpenAI` instance from a Provider value.

  `extra_attrs` is a keyword or map of overrides merged into the base
  config — strategies use this to set `json_response`, `json_schema`,
  `tools`, `tool_choice`, etc.

  Non-streaming by default. Set `config :defacto_ai, :chat_stream, true`
  to enable streaming globally — useful when a gateway (e.g. Heroku
  Inference) rejects long-running non-streamed chat completions.
  Per-call `stream: ...` overrides win over the Application env.

  LangChain reassembles streamed deltas so the final `last_message`
  looks identical to a non-streamed response.
  """
  @spec build_chat_model(any(), keyword() | map()) :: ChatOpenAI.t()
  def build_chat_model(provider, extra_attrs \\ []) do
    base = %{
      endpoint: endpoint(provider),
      api_key: Provider.api_key(provider),
      model: Provider.model(provider),
      receive_timeout: @default_receive_timeout,
      stream: stream_default()
    }

    ChatOpenAI.new!(Map.merge(base, Map.new(extra_attrs)))
  end

  defp stream_default do
    Application.get_env(:defacto_ai, :chat_stream, false)
  end

  @doc """
  Build an `LLMChain` from a chat model and a list of messages.

  Messages may be raw `%LangChain.Message{}` structs or simple maps with
  `:role` and `:content` keys (the shape the existing call sites use).
  """
  @spec build_chain(ChatOpenAI.t(), [message_input()], keyword()) :: LLMChain.t()
  def build_chain(%ChatOpenAI{} = chat_model, messages, opts \\ []) do
    chain_attrs =
      %{llm: chat_model, verbose: Keyword.get(opts, :verbose, false)}
      |> maybe_put_custom_context(opts)

    LLMChain.new!(chain_attrs)
    |> LLMChain.add_messages(Enum.map(messages, &to_message/1))
  end

  defp maybe_put_custom_context(attrs, opts) do
    case Keyword.get(opts, :custom_context) do
      nil -> attrs
      context -> Map.put(attrs, :custom_context, context)
    end
  end

  defp endpoint(provider) do
    base = Provider.base_url(provider) || ""
    path = Provider.api_path(provider) || ""
    String.trim_trailing(base, "/") <> path
  end

  defp to_message(%Message{} = m), do: m
  defp to_message(%{role: role, content: content}), do: build_message(role, content)
  defp to_message(%{"role" => role, "content" => content}), do: build_message(role, content)

  defp build_message(role, content) when role in [:user, "user"], do: Message.new_user!(content)

  defp build_message(role, content) when role in [:system, "system"],
    do: Message.new_system!(content)

  defp build_message(role, content) when role in [:assistant, "assistant"],
    do: Message.new_assistant!(content)
end
