defmodule DefactoAI.PromptRenderer do
  @moduledoc """
  Renders prompt templates by performing simple `{{key}}` placeholder
  substitution.

  Templates are looked up in the host application's `priv/<dir>/`
  directory. Configure the host app once:

      # config/config.exs
      config :defacto_ai, prompts_app: :my_app
      # or, if you want prompts in a different subdirectory:
      config :defacto_ai, prompts_app: :my_app, prompts_dir: "ai_prompts"

  Then call from anywhere:

      DefactoAI.PromptRenderer.render("session_summary.md", %{user: "Alice"})

  Absolute paths bypass the lookup and are read as-is, which is handy
  for tests and for hosts that bundle prompts outside `priv/`.

  Substitution is intentionally minimal — `{{key}}` becomes
  `to_string(vars[key])`. For richer templating (loops, conditionals,
  HTML escaping), use EEx directly.
  """

  @default_dir "prompts"

  @doc """
  Render `template` with `vars`.

  `template` is one of:
    * an absolute path (`"/abs/path/to/file.md"`) — read directly,
    * a binary template name (`"session_summary.md"`) — looked up in
      `priv/<prompts_dir>/<template>` of the configured host app,
    * a binary literal containing the template body itself when it
      starts with the `inline:` prefix (`"inline:Hello {{name}}!"`).
  """
  @spec render(String.t(), map() | keyword()) :: String.t()
  def render(template, vars) when is_binary(template) do
    template
    |> read_template()
    |> interpolate(vars)
  end

  defp read_template("inline:" <> body), do: body

  defp read_template("/" <> _ = absolute_path), do: File.read!(absolute_path)

  defp read_template(name) when is_binary(name) do
    case Application.get_env(:defacto_ai, :prompts_app) do
      nil ->
        raise """
        DefactoAI.PromptRenderer: no :prompts_app configured.

        Set the host app for prompt lookups:

            config :defacto_ai, prompts_app: :my_app

        Or pass an absolute path / inline template directly to render/2.
        """

      app ->
        dir = Application.get_env(:defacto_ai, :prompts_dir, @default_dir)

        app
        |> :code.priv_dir()
        |> Path.join(dir)
        |> Path.join(name)
        |> File.read!()
    end
  end

  defp interpolate(template, vars) do
    Enum.reduce(vars, template, fn {key, val}, acc ->
      String.replace(acc, "{{#{key}}}", to_string(val))
    end)
  end
end
