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

  alias DefactoAI.Telemetry

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
          | {:telemetry_tags, map()}
          | {:role, role()}
          | {:max_validation_retries, non_neg_integer()}
          | {:strategies, [module()]}
          | {:validation_context, map()}
          | {atom(), term()}
        ]

  @callback complete_structured(
              schema_module :: module(),
              messages :: [message()],
              opts :: opts()
            ) ::
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
    {tags, opts} = pop_tags(default_role(opts, :llm))
    role = Keyword.get(opts, :role)

    :telemetry.span(
      [:defacto_ai, :complete_structured],
      %{schema: schema_module, role: role, tags: tags},
      fn ->
        {result, context} =
          Telemetry.track(fn -> impl().complete_structured(schema_module, messages, opts) end,
            provider: Keyword.get(opts, :provider)
          )

        {result, stop_metadata(context, result, role, tags, %{schema: schema_module})}
      end
    )
  end

  @doc """
  Generate a plain-text chat completion (non-streaming). Returns the
  assembled assistant message content as a binary.
  """
  @spec complete_chat([message()], opts()) :: {:ok, binary()} | {:error, term()}
  def complete_chat(messages, opts \\ []) do
    {tags, opts} = pop_tags(default_role(opts, :llm))
    role = Keyword.get(opts, :role)

    :telemetry.span(
      [:defacto_ai, :complete_chat],
      %{role: role, tags: tags},
      fn ->
        {result, context} =
          Telemetry.track(fn -> impl().complete_chat(messages, opts) end,
            provider: Keyword.get(opts, :provider)
          )

        {result, stop_metadata(context, result, role, tags, %{})}
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
    {tags, opts} = pop_tags(default_role(opts, :llm))
    role = Keyword.get(opts, :role)
    event = [:defacto_ai, :stream_chat]
    started_at = System.monotonic_time()

    :telemetry.execute(
      event ++ [:start],
      %{monotonic_time: started_at, system_time: System.system_time()},
      %{role: role, tags: tags}
    )

    try do
      Telemetry.track(fn -> impl().stream_chat(messages, opts) end,
        provider: Keyword.get(opts, :provider)
      )
    catch
      kind, reason ->
        stacktrace = __STACKTRACE__

        :telemetry.execute(
          event ++ [:exception],
          %{duration: System.monotonic_time() - started_at},
          %{role: role, tags: tags, kind: kind, reason: reason, stacktrace: stacktrace}
        )

        :erlang.raise(kind, reason, stacktrace)
    else
      {{:ok, stream}, context} ->
        {:ok, observe_stream(stream, context, started_at, role, tags)}

      {error, context} ->
        stop_stream(context, started_at, role, tags, :error)
        error
    end
  end

  # Wraps `stream` so the stop (or exception) event fires when the consumer
  # finishes, halts early, or the underlying enumerable raises. Elements are
  # passed through untouched.
  defp observe_stream(stream, context, started_at, role, tags) do
    Stream.resource(
      fn ->
        {:suspended, nil, continuation} =
          Enumerable.reduce(stream, {:suspend, nil}, fn element, _ -> {:suspend, element} end)

        %{continuation: continuation, result: :ok, finished: false, ref: make_ref()}
      end,
      fn
        %{finished: true} = state ->
          {:halt, state}

        %{continuation: continuation, ref: ref} = state ->
          try do
            continuation.({:cont, nil})
          catch
            kind, reason ->
              stacktrace = __STACKTRACE__
              emit_stream_exception(context, started_at, role, tags, kind, reason, stacktrace)
              # Stream.resource still runs the cleanup function after a
              # raise; this tells it the exception event already went out.
              Process.put({:defacto_stream_raised, ref}, true)
              :erlang.raise(kind, reason, stacktrace)
          else
            {:suspended, {:error, _} = element, continuation} ->
              {[element], %{state | continuation: continuation, result: :error}}

            {:suspended, element, continuation} ->
              {[element], %{state | continuation: continuation}}

            {_done_or_halted, _acc} ->
              {:halt, %{state | finished: true}}
          end
      end,
      fn %{continuation: continuation, ref: ref} = state ->
        if Process.delete({:defacto_stream_raised, ref}) do
          :ok
        else
          # Let the underlying stream run its own cleanup (e.g. stop the request).
          unless state.finished, do: continuation.({:halt, nil})
          stop_stream(context, started_at, role, tags, state.result)
        end
      end
    )
  end

  defp stop_stream(context, started_at, role, tags, result) do
    metadata = stop_metadata(context, result, role, tags, %{})

    :telemetry.execute(
      [:defacto_ai, :stream_chat, :stop],
      %{duration: System.monotonic_time() - started_at, monotonic_time: System.monotonic_time()},
      metadata
    )
  end

  defp emit_stream_exception(context, started_at, role, tags, kind, reason, stacktrace) do
    metadata =
      context
      |> stop_metadata(:error, role, tags, %{})
      |> Map.merge(%{kind: kind, reason: reason, stacktrace: stacktrace})

    :telemetry.execute(
      [:defacto_ai, :stream_chat, :exception],
      %{duration: System.monotonic_time() - started_at},
      metadata
    )
  end

  # `:telemetry_tags` is for telemetry handlers only. It is removed here so
  # it can never reach a provider request or a schema changeset.
  defp pop_tags(opts) do
    {tags, opts} = Keyword.pop(opts, :telemetry_tags, %{})
    {Map.new(tags), opts}
  end

  defp stop_metadata(context, result, role, tags, extra) do
    context
    |> Telemetry.info()
    |> Map.merge(%{result: Telemetry.result_status(result), role: role, tags: tags})
    |> Map.merge(extra)
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
