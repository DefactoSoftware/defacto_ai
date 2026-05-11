defmodule DefactoAI.Repo do
  @moduledoc false
  # Internal helper for fetching the host application's configured Repo.
  # Library code uses `DefactoAI.Repo.get!/0` instead of aliasing a fixed
  # repo module so that any consumer can wire their own repo through:
  #
  #     config :defacto_ai, repo: MyApp.Repo

  @spec get!() :: module()
  def get! do
    Application.get_env(:defacto_ai, :repo) ||
      raise """
      DefactoAI: no Repo configured.

      Set the host app's repo in config:

          config :defacto_ai, repo: MyApp.Repo
      """
  end
end
