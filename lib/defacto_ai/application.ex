defmodule DefactoAI.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children =
      [
        DefactoAI.Embeddings.RateLimiter
      ]
      |> Enum.filter(&Code.ensure_loaded?/1)

    Supervisor.start_link(children, strategy: :one_for_one, name: DefactoAI.Supervisor)
  end
end
