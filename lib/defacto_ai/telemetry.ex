defmodule DefactoAI.Telemetry do
  @moduledoc """
  Collects per-call details (provider, model, token usage) that end up in
  the metadata of the library's telemetry events.

  A call opens a context with `track/1`. Code running inside it (provider
  resolution, the repair loop, the HTTP clients) reports what it learns
  through `record_provider/1` and `record_usage/1,2`. When the call ends,
  `info/1` turns the context into the `provider`, `model` and `usage`
  metadata keys.

  The context lives in the process dictionary of the calling process.
  Usage itself is stored in an `:atomics` array, so processes that were
  handed the context (see `current/0` and `attach/1`) can report usage
  too. Streaming uses this: the request runs in a task, the telemetry
  event is emitted by whichever process finishes the stream.

  This module is public so the library's own modules can share it. Hosts
  normally only read the resulting event metadata and do not call it.
  """

  @key :defacto_ai_telemetry

  @typedoc "Token counts for one call. A value is `nil` when the provider did not report it."
  @type usage :: %{input_tokens: non_neg_integer() | nil, output_tokens: non_neg_integer() | nil}

  @typedoc "Opaque handle for one call; see `track/1`."
  @opaque context :: map()

  @typedoc "Per-call details derived from a context."
  @type info :: %{provider: String.t() | nil, model: String.t() | nil, usage: usage()}

  @doc """
  Run `fun` inside a fresh context. Returns `{result, context}`.

  The previous context of the calling process (if any) is restored
  afterwards, so calls can nest.
  """
  @spec track((-> result), keyword()) :: {result, context()} when result: term()
  def track(fun, opts \\ []) when is_function(fun, 0) do
    previous = Process.get(@key)
    context = %{usage: :atomics.new(2, signed: false), provider: nil, model: nil}
    Process.put(@key, context)

    case Keyword.get(opts, :provider) do
      nil -> :ok
      provider -> record_provider(provider)
    end

    try do
      result = fun.()
      {result, Process.get(@key)}
    after
      restore(previous)
    end
  end

  @doc "The context of the calling process, or `nil` outside `track/1`."
  @spec current() :: context() | nil
  def current, do: Process.get(@key)

  @doc """
  Make `context` the context of the calling process (for example inside a
  task spawned during a tracked call). `nil` is ignored.
  """
  @spec attach(context() | nil) :: :ok
  def attach(nil), do: :ok

  def attach(context) do
    Process.put(@key, context)
    :ok
  end

  @doc "Remember which provider and model a call resolved to."
  @spec record_provider(term()) :: :ok
  def record_provider(provider) do
    update(fn context ->
      %{
        context
        | provider: label(DefactoAI.Provider.provider_type(provider)),
          model: label(DefactoAI.Provider.model(provider))
      }
    end)
  end

  @doc """
  Add the token usage of one provider round to the call's total.

  Accepts a `LangChain.TokenUsage`, `nil` (ignored), or the two counts.
  A count that is `nil` leaves that side of the total untouched, so a total
  stays `nil` until a provider reports it.
  """
  @spec record_usage(LangChain.TokenUsage.t() | nil) :: :ok
  def record_usage(nil), do: :ok

  def record_usage(%LangChain.TokenUsage{input: input, output: output}),
    do: record_usage(input, output)

  @spec record_usage(non_neg_integer() | nil, non_neg_integer() | nil) :: :ok
  def record_usage(input, output) do
    case current() do
      %{usage: ref} ->
        add(ref, 1, input)
        add(ref, 2, output)

      nil ->
        :ok
    end
  end

  @doc "Record the usage reported on a `LangChain.Message` (or message delta), if any."
  @spec record_message_usage(term()) :: :ok
  def record_message_usage(%{metadata: %{usage: %LangChain.TokenUsage{} = usage}}),
    do: record_usage(usage)

  def record_message_usage(_), do: :ok

  @doc "Turn a context into the `provider`, `model` and `usage` metadata."
  @spec info(context() | nil) :: info()
  def info(nil), do: %{provider: nil, model: nil, usage: %{input_tokens: nil, output_tokens: nil}}

  def info(%{usage: ref} = context) do
    %{
      provider: context.provider,
      model: context.model,
      usage: %{input_tokens: read(ref, 1), output_tokens: read(ref, 2)}
    }
  end

  @doc "`:ok` for `{:ok, _}` results, `:error` for everything else."
  @spec result_status(term()) :: :ok | :error
  def result_status({:ok, _}), do: :ok
  def result_status(_), do: :error

  # Counts are stored as `count + 1`; 0 means "never reported".
  defp add(_ref, _slot, nil), do: :ok

  defp add(ref, slot, count) when is_integer(count) and count >= 0 do
    if :atomics.get(ref, slot) == 0 do
      :atomics.put(ref, slot, count + 1)
    else
      :atomics.add(ref, slot, count)
    end
  end

  defp add(_ref, _slot, _other), do: :ok

  defp read(ref, slot) do
    case :atomics.get(ref, slot) do
      0 -> nil
      stored -> stored - 1
    end
  end

  defp update(fun) do
    case current() do
      nil -> :ok
      context -> Process.put(@key, fun.(context))
    end

    :ok
  end

  defp restore(nil), do: Process.delete(@key)
  defp restore(previous), do: Process.put(@key, previous)

  defp label(nil), do: nil
  defp label(value) when is_binary(value), do: value
  defp label(value), do: to_string(value)
end
