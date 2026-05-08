defmodule DefactoAI.Client do
  @moduledoc """
  Public entry point for taking structured / chat / streaming output from
  an LLM.

  Defined as a behaviour with two implementations: the production
  `DefactoAI.Client.LangChain` and the test-only `DefactoAI.Client.Stub`.
  Pick the implementation via:

      config :defacto_ai, :client, DefactoAI.Client.LangChain

  Call sites only ever talk to `DefactoAI.Client.complete_structured/3`,
  `complete_chat/2`, or `stream_chat/2`. Provider lookup, strategy
  fallback, JSON repair and validation retry all happen behind this
  facade.

  ## Provider resolution

  A call resolves the provider in this order:

    1. Explicit `provider:` option — any value implementing the
       `DefactoAI.Provider` protocol is used as-is.
    2. Otherwise, a `role:` option is used (defaulting to `:llm` for
       chat-style functions). The role is passed to the host's
       `:provider_resolver` callback configured under `:defacto_ai`:

           config :defacto_ai, provider_resolver: &MyApp.AI.resolve_provider/1

       The resolver is given the role atom and must return a struct
       implementing `DefactoAI.Provider`, or `nil` if no provider is
       configured for that role.

  The library defines three canonical roles — `:llm`, `:embedding`,
  `:summary` — but the resolver callback is opaque to it: hosts may
  define their own role atoms and pass them through `role:` if they
  wish.
  """

  @typedoc """
  Either a fully-formed `LangChain.Message`, a map with atom-keyed
  `:role` and `:content`, or a map with string-keyed `"role"` and
  `"content"`.
  """
  @type message :: map() | LangChain.Message.t()

  @typedoc "Canonical roles understood by the default `Embeddings`/`Client` callers."
  @type role :: :llm | :summary | :embedding | atom()

  @typedoc "Anything implementing `DefactoAI.Provider`."
  @type provider :: any()

  @type opts :: [
          {:provider, provider()}
          | {:role, role()}
          | {:max_validation_retries, non_neg_integer()}
          | {:strategies, [module()]}
          | {:validation_context, map()}
          | {atom(), term()}
        ]

  @callback complete_structured(schema_module :: module(), messages :: [message()], opts :: opts()) ::
              {:ok, struct()} | {:error, term()}

  @callback complete_chat(messages :: [message()], opts :: opts()) ::
              {:ok, binary()} | {:error, term()}

  @callback stream_chat(messages :: [message()], opts :: opts()) ::
              {:ok, Enumerable.t()} | {:error, term()}

  @doc """
  Generate a structured response that's been cast and validated through
  `schema_module`'s changeset.

  Defaults `role:` to `:llm` if neither `provider:` nor `role:` is given.
  Resolves the implementation via
  `Application.get_env(:defacto_ai, :client, DefactoAI.Client.LangChain)`.
  """
  @spec complete_structured(module(), [message()], opts()) :: {:ok, struct()} | {:error, term()}
  def complete_structured(schema_module, messages, opts \\ []) do
    opts = default_role(opts, :llm)

    :telemetry.span(
      [:defacto_ai, :complete_structured],
      %{schema: schema_module, role: Keyword.get(opts, :role)},
      fn ->
        result = impl().complete_structured(schema_module, messages, opts)
        {result, %{schema: schema_module}}
      end
    )
  end

  @doc """
  Generate a plain-text chat completion (non-streaming). Returns the
  assembled assistant message content as a binary.
  """
  @spec complete_chat([message()], opts()) :: {:ok, binary()} | {:error, term()}
  def complete_chat(messages, opts \\ []) do
    opts = default_role(opts, :llm)

    :telemetry.span(
      [:defacto_ai, :complete_chat],
      %{role: Keyword.get(opts, :role)},
      fn ->
        result = impl().complete_chat(messages, opts)
        {result, %{}}
      end
    )
  end

  @doc """
  Generate a chat completion as a stream of content chunks.

  Returns `{:ok, stream}` where `stream` yields binary chunks of the
  assistant's reply. The stream may yield `{:error, reason}` as a
  terminating element if the upstream call fails mid-stream — consumers
  should pattern-match each element to decide whether to continue.
  """
  @spec stream_chat([message()], opts()) :: {:ok, Enumerable.t()} | {:error, term()}
  def stream_chat(messages, opts \\ []) do
    opts = default_role(opts, :llm)
    impl().stream_chat(messages, opts)
  end

  defp default_role(opts, role) do
    if Keyword.has_key?(opts, :provider) or Keyword.has_key?(opts, :role) do
      opts
    else
      Keyword.put(opts, :role, role)
    end
  end

  defp impl do
    Application.get_env(:defacto_ai, :client, DefactoAI.Client.LangChain)
  end
end
