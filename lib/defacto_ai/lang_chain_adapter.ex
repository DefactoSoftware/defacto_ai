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

  @doc """
  Like `build_chat_model/2`, but also merges the caller's per-call chat
  model attributes from `opts[:chat_model]` (see `chat_model_attrs/1`).

  `extra_attrs` are the strategy's own attributes (`tool_choice`,
  `json_response`, `stream: false`, ...) and always win over the caller's,
  so a host cannot accidentally undo what a strategy relies on.
  """
  @spec build_chat_model(any(), keyword() | map(), keyword()) :: ChatOpenAI.t()
  def build_chat_model(provider, extra_attrs, opts) when is_list(opts) do
    attrs = Map.merge(chat_model_attrs(opts), Map.new(extra_attrs))
    build_chat_model(provider, attrs)
  end

  @doc """
  The caller's per-call `ChatOpenAI` attributes: `opts[:chat_model]` as a
  keyword list or map (e.g. `chat_model: [max_tokens: 8_000]`), normalised
  to a map. Missing or `nil` yields `%{}`.

  Some gateways default `max_tokens` too low for long structured answers;
  a truncated tool-call arguments JSON then fails to parse and cascades
  through every strategy. Hosts can raise it per call with this option.
  """
  @spec chat_model_attrs(keyword()) :: map()
  def chat_model_attrs(opts) when is_list(opts) do
    case Keyword.get(opts, :chat_model) do
      nil -> %{}
      attrs when is_list(attrs) or is_map(attrs) -> Map.new(attrs)
    end
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
