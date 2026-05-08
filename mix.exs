defmodule DefactoAI.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :defacto_ai,
      version: @version,
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "Shared LLM, embeddings and similarity-search building blocks for Defacto apps.",
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {DefactoAI.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:langchain, "~> 0.8"},
      {:req, "~> 0.5"},
      {:ecto_sql, "~> 3.12"},
      {:postgrex, "~> 0.20"},
      {:pgvector, "~> 0.3"},
      {:telemetry, "~> 1.2"},
      {:jason, "~> 1.4"},
      # Test
      {:plug, "~> 1.16", only: :test},
      {:bypass, "~> 2.1", only: :test},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      maintainers: ["Defacto Software"],
      licenses: ["Proprietary"],
      links: %{}
    ]
  end
end
