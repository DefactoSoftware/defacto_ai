defmodule DefactoAI.Embeddings.RateLimiter do
  @moduledoc """
  Simple rate limiter for embedding API calls.

  Uses ETS to track request timestamps in a sliding window.
  Blocks callers when the rate limit is exceeded until capacity is
  available.

  Started under `DefactoAI.Application`'s supervisor when the module is
  compiled into the host app.

  Configure the per-minute limit:

      config :defacto_ai, embedding_rate_limit: 450
  """

  use GenServer

  @table_name :defacto_ai_embedding_rate_limiter
  @default_rate_limit 450
  @window_ms 60_000

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Acquires capacity for a request, blocking until available.

  This atomically checks capacity AND records the request to prevent
  race conditions with concurrent workers.

  Returns `:ok` when the caller can proceed.
  """
  def acquire do
    GenServer.call(__MODULE__, :acquire, :infinity)
  end

  @doc """
  Returns the configured rate limit per minute.
  """
  def rate_limit do
    Application.get_env(:defacto_ai, :embedding_rate_limit, @default_rate_limit)
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    table = :ets.new(@table_name, [:named_table, :ordered_set, :public])
    {:ok, %{table: table}}
  end

  @impl true
  def handle_call(:acquire, _from, state) do
    wait_until_capacity_available()
    # Record the request atomically after confirming capacity
    now = System.monotonic_time(:millisecond)
    :ets.insert(@table_name, {now, true})
    {:reply, :ok, state}
  end

  defp wait_until_capacity_available do
    cleanup_old_requests()
    current_count = :ets.info(@table_name, :size)
    limit = rate_limit()

    if current_count >= limit do
      # Calculate how long to wait for the oldest request to expire
      case :ets.first(@table_name) do
        :"$end_of_table" ->
          :ok

        oldest_timestamp ->
          now = System.monotonic_time(:millisecond)
          wait_time = max(0, oldest_timestamp + @window_ms - now + 100)

          if wait_time > 0 do
            Process.sleep(wait_time)
            wait_until_capacity_available()
          end
      end
    end
  end

  defp cleanup_old_requests do
    cutoff = System.monotonic_time(:millisecond) - @window_ms

    :ets.select_delete(@table_name, [
      {{:"$1", :_}, [{:<, :"$1", cutoff}], [true]}
    ])
  end
end
