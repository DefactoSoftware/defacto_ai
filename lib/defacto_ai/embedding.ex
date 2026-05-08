defmodule DefactoAI.Embedding do
  @moduledoc """
  Schema for storing vector embeddings of embeddable content.

  The host application owns the underlying database table — install it
  with `mix defacto_ai.gen.migration` (which generates a migration that
  creates the pgvector extension and the `embeddings` table with the
  schema this module expects).

  Reads and writes go through the host's configured Repo:

      config :defacto_ai, repo: MyApp.Repo
  """

  use Ecto.Schema

  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "embeddings" do
    field :embeddable_type, :string
    field :embeddable_id, :string
    field :embeddable_title, :string
    field :embedding, Pgvector.Ecto.Vector
    field :page_number, :integer
    field :chunk_index, :integer
    field :chunk_text, :string

    timestamps()
  end

  @doc """
  Creates a changeset for an embedding.
  """
  def changeset(embedding, attrs) do
    embedding
    |> cast(attrs, [
      :embeddable_type,
      :embeddable_id,
      :embeddable_title,
      :embedding,
      :page_number,
      :chunk_index,
      :chunk_text
    ])
    |> validate_required([:embeddable_type, :embeddable_id, :embeddable_title, :embedding])
  end

  @doc """
  Returns the nearest neighbours to the given embedding vector.

  ## Options
    * `:limit` - Maximum number of results to return (default: 10)
    * `:embeddable_types` - List of embeddable types to filter by (optional)
  """
  def nearest_neighbors(embedding, opts \\ []) do
    limit = Keyword.get(opts, :limit, 10)
    embeddable_types = Keyword.get(opts, :embeddable_types, nil)

    query =
      from(e in __MODULE__,
        order_by: fragment("embedding <=> ?", type(^embedding, Pgvector.Ecto.Vector)),
        limit: ^limit,
        select: %{
          id: e.id,
          embeddable_type: e.embeddable_type,
          embeddable_id: e.embeddable_id,
          embeddable_title: e.embeddable_title,
          chunk_index: e.chunk_index,
          chunk_text: e.chunk_text,
          distance: fragment("embedding <=> ?", type(^embedding, Pgvector.Ecto.Vector))
        }
      )

    query =
      if embeddable_types do
        type_strings = Enum.map(embeddable_types, &to_string/1)
        where(query, [e], e.embeddable_type in ^type_strings)
      else
        query
      end

    DefactoAI.Repo.get!().all(query)
  end

  @doc """
  Deletes all embeddings for a given embeddable.
  """
  def delete_for_embeddable(embeddable_type, embeddable_id) do
    from(e in __MODULE__,
      where: e.embeddable_type == ^to_string(embeddable_type),
      where: e.embeddable_id == ^to_string(embeddable_id)
    )
    |> DefactoAI.Repo.get!().delete_all()
  end
end
