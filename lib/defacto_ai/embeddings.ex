defmodule DefactoAI.Embeddings do
  @moduledoc """
  High-level embedding orchestration: chunk content if necessary, embed
  via the HTTP client, store rows in the host's `embeddings` table.

  Public surface:

    * `embed/2`, `embed_batch/2` — pure embedding generation, delegated
      to `DefactoAI.Embeddings.Client`. Use these when the caller wants
      raw vectors without persistence.
    * `generate/2` — embed an embeddable struct end-to-end (chunk if
      `chunked?/0`, delete existing rows, insert new ones).
    * `store/4` — write a single embedding row.
    * `delete_for/1` — delete every row for an embeddable.
    * `has_for?/1` — check whether any rows exist.
    * `nearest/2` — wrapper around `DefactoAI.Embedding.nearest_neighbors/2`.

  Async scheduling (e.g. via Oban), embeddable-module registries and
  reporting queries are intentionally left to the host application —
  those are domain concerns that don't belong in the library.
  """

  import Ecto.Query

  alias DefactoAI.Embedding
  alias DefactoAI.Embeddings.{Chunker, Client}

  # Stays in sync with the upstream API limit. The chunker uses 1200 to
  # leave buffer for encoding overhead; we cap single (unchunked) inputs
  # at the same limit so non-chunkable embeddables don't trip provider
  # limits.
  @max_content_length 1200

  @doc """
  Generate an embedding for a single text.

  Delegates to `DefactoAI.Embeddings.Client.embed/2`. Wrapped in a
  telemetry span (`[:defacto_ai, :embed, :start | :stop | :exception]`)
  so hosts can observe per-request durations.
  """
  def embed(text, opts \\ []) do
    :telemetry.span([:defacto_ai, :embed], %{}, fn ->
      {Client.embed(text, opts), %{}}
    end)
  end

  @doc """
  Generate embeddings for multiple texts.
  """
  def embed_batch(texts, opts \\ []) do
    :telemetry.span([:defacto_ai, :embed_batch], %{count: length(texts)}, fn ->
      {Client.embed_batch(texts, opts), %{count: length(texts)}}
    end)
  end

  @doc """
  Generate and store embeddings for an embeddable struct.

  The struct must `use DefactoAI.Embeddings.Embeddable` (or otherwise
  implement the behaviour) so `embedding_content/1`, `embeddable_type/0`
  and `chunked?/0` are callable on its module.

  Returns:

    * `{:ok, %DefactoAI.Embedding{}}` for unchunked content.
    * `{:ok, count}` for chunked content (count = number of stored rows).
    * `{:error, reason}` on the first failure.
  """
  def generate(embeddable, opts \\ []) do
    module = embeddable.__struct__
    content = module.embedding_content(embeddable)
    title = get_title(embeddable)

    if module.chunked?() do
      generate_chunked_embedding(embeddable, content, title, opts)
    else
      generate_single_embedding(embeddable, content, title, opts)
    end
  end

  defp generate_single_embedding(embeddable, content, title, opts) do
    truncated_content = String.slice(content || "", 0, @max_content_length)

    case Client.embed(truncated_content, opts) do
      {:ok, embedding} ->
        delete_for(embeddable)
        store(embeddable, title, embedding)

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp generate_chunked_embedding(embeddable, content, title, opts) do
    chunks = Chunker.chunk(content)
    texts = Enum.map(chunks, & &1.content)

    case Client.embed_batch(texts, opts) do
      {:ok, embeddings} ->
        delete_for(embeddable)

        results =
          Enum.zip(chunks, embeddings)
          |> Enum.map(fn {chunk, embedding} ->
            store(embeddable, title, embedding,
              chunk_index: chunk.index,
              chunk_text: chunk.content
            )
          end)

        errors = Enum.filter(results, &match?({:error, _}, &1))

        if Enum.empty?(errors) do
          {:ok, length(results)}
        else
          {:error, {:partial_failure, errors}}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Stores an embedding for an embeddable record.

  ## Options
    * `:chunk_index` - Index of the chunk (for chunked content)
    * `:chunk_text` - The text content of the chunk
    * `:page_number` - Page number (for paginated content like PDFs)
  """
  def store(embeddable, title, embedding, opts \\ []) do
    embeddable_type = embeddable.__struct__.embeddable_type()

    attrs = %{
      embeddable_type: to_string(embeddable_type),
      embeddable_id: to_string(embeddable.id),
      embeddable_title: title,
      embedding: embedding,
      chunk_index: Keyword.get(opts, :chunk_index),
      chunk_text: Keyword.get(opts, :chunk_text),
      page_number: Keyword.get(opts, :page_number)
    }

    %Embedding{}
    |> Embedding.changeset(attrs)
    |> DefactoAI.Repo.get!().insert()
  end

  @doc """
  Deletes existing embeddings for an embeddable record.
  """
  def delete_for(embeddable) do
    embeddable_type = embeddable.__struct__.embeddable_type()
    Embedding.delete_for_embeddable(embeddable_type, embeddable.id)
  end

  @doc """
  Checks if an embeddable record has embeddings.
  """
  def has_for?(embeddable) do
    embeddable_type = embeddable.__struct__.embeddable_type()

    from(e in Embedding,
      where: e.embeddable_type == ^to_string(embeddable_type),
      where: e.embeddable_id == ^to_string(embeddable.id),
      select: count(e.id)
    )
    |> DefactoAI.Repo.get!().one()
    |> Kernel.>(0)
  end

  @doc """
  Returns nearest neighbours to a pre-computed embedding vector.

  Wrapper around `DefactoAI.Embedding.nearest_neighbors/2` for callers
  that want a stable public surface.
  """
  defdelegate nearest(embedding, opts \\ []), to: Embedding, as: :nearest_neighbors

  defp get_title(embeddable) do
    cond do
      present?(Map.get(embeddable, :title)) -> embeddable.title
      present?(Map.get(embeddable, :name)) -> embeddable.name
      true -> "#{embeddable.__struct__} #{embeddable.id}"
    end
  end

  defp present?(nil), do: false
  defp present?(""), do: false
  defp present?(_), do: true
end
