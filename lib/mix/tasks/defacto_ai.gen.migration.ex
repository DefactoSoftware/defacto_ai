defmodule Mix.Tasks.DefactoAi.Gen.Migration do
  @shortdoc "Generates the embeddings-table migration for the host app"

  @moduledoc """
  Generates a migration that installs the pgvector extension and the
  `embeddings` table that `DefactoAI.Embedding` expects.

      mix defacto_ai.gen.migration

  Output goes to `priv/repo/migrations/<timestamp>_create_defacto_ai_embeddings.exs`
  in the *host* application (not in this library). Pass `--repo` to use
  a non-default Repo:

      mix defacto_ai.gen.migration --repo MyApp.OtherRepo

  Pass `--vector-size` to override the default 1024 if your embeddings
  provider returns a different dimension:

      mix defacto_ai.gen.migration --vector-size 1536

  After generation, run `mix ecto.migrate` as usual.
  """

  use Mix.Task

  import Mix.Generator

  @switches [
    repo: [:keep, :string],
    vector_size: :integer
  ]

  @impl true
  def run(args) do
    no_umbrella!()
    Mix.Task.run("app.config")

    {opts, _argv, _errors} = OptionParser.parse(args, switches: @switches)

    repo = pick_repo(opts)
    vector_size = Keyword.get(opts, :vector_size, 1024)

    ensure_repo!(repo)

    path = Path.join(source_repo_priv(repo), "migrations")
    File.mkdir_p!(path)

    timestamp = timestamp()
    file = Path.join(path, "#{timestamp}_create_defacto_ai_embeddings.exs")
    module = "#{inspect(repo)}.Migrations.CreateDefactoAiEmbeddings"

    create_file(file, migration_template(module: module, vector_size: vector_size))

    Mix.shell().info("""

    Run `mix ecto.migrate` to apply.
    """)
  end

  defp pick_repo(opts) do
    case Keyword.get_values(opts, :repo) do
      [] ->
        case Mix.Ecto.parse_repo([]) do
          [repo | _] -> repo
          [] -> Mix.raise("No Ecto repos found.")
        end

      [name] ->
        Module.concat([name])

      [_ | _] ->
        Mix.raise("--repo can only be passed once")
    end
  end

  defp ensure_repo!(repo) do
    case Code.ensure_compiled(repo) do
      {:module, _} ->
        :ok

      {:error, reason} ->
        Mix.raise(
          "Could not load #{inspect(repo)}: #{inspect(reason)}. " <>
            "Make sure --repo points at a real Ecto.Repo module."
        )
    end
  end

  defp source_repo_priv(repo) do
    config = repo.config()
    priv = config[:priv] || "priv/#{repo |> Module.split() |> List.last() |> Macro.underscore()}"
    Path.join(File.cwd!(), priv)
  end

  defp no_umbrella! do
    if Mix.Project.umbrella?() do
      Mix.raise(
        "mix defacto_ai.gen.migration must be run inside a single OTP app, " <>
          "not the umbrella root."
      )
    end
  end

  defp timestamp do
    {{y, m, d}, {hh, mm, ss}} = :calendar.universal_time()
    "#{y}#{pad(m)}#{pad(d)}#{pad(hh)}#{pad(mm)}#{pad(ss)}"
  end

  defp pad(i) when i < 10, do: "0#{i}"
  defp pad(i), do: to_string(i)

  embed_template(:migration, """
  defmodule <%= @module %> do
    use Ecto.Migration

    def change do
      execute(
        "CREATE EXTENSION IF NOT EXISTS vector",
        "DROP EXTENSION IF EXISTS vector"
      )

      create table(:embeddings, primary_key: false) do
        add :id, :binary_id, primary_key: true
        add :embeddable_type, :string, null: false
        add :embeddable_id, :binary_id, null: false
        add :embeddable_title, :string, null: false
        add :embedding, :vector, size: <%= @vector_size %>, null: false
        add :page_number, :integer
        add :chunk_index, :integer
        add :chunk_text, :text

        timestamps()
      end

      create index(:embeddings, [:embeddable_type, :embeddable_id])

      create index(:embeddings, ["embedding vector_cosine_ops"],
               using: :hnsw,
               name: :embeddings_embedding_hnsw_index
             )
    end
  end
  """)
end
