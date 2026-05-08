defmodule DefactoAI.ErrorReporter do
  @moduledoc """
  Optional error-reporting hook.

  Library code that wants to report a non-fatal anomaly (an upstream
  HTTP failure, a transient API error worth grouping in Sentry) calls
  `report/2`. The host application configures the actual reporter:

      config :defacto_ai, error_reporter: &MyApp.AI.report_error/2

  The reporter receives a stable string `title` (e.g.
  `"Embedding API error"`) and an `extra` map. Stable titles are
  important for grouping in tools like Sentry.

  When no reporter is configured, errors only land in `Logger.warning/1`
  — the call site that triggered `report/2` is expected to log
  separately for that reason.
  """

  @type extra :: map()
  @type reporter :: (binary(), extra() -> :ok)

  @spec report(binary(), extra()) :: :ok
  def report(title, extra) when is_binary(title) and is_map(extra) do
    case Application.get_env(:defacto_ai, :error_reporter) do
      fun when is_function(fun, 2) ->
        fun.(title, extra)
        :ok

      _ ->
        :ok
    end
  end
end
