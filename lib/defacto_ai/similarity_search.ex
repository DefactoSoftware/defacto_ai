defmodule DefactoAI.SimilaritySearch do
  @moduledoc """
  Performs similarity search using vector embeddings with decay-based
  scoring.

  The decay algorithm groups results by `embeddable_id` and applies
  exponential decay to chunk scores, giving higher weight to the
  best-matching chunks while still considering additional matches.
  """

  alias DefactoAI.Embedding
  alias DefactoAI.Embeddings.Client, as: EmbeddingClient

  @default_limit 20
  @default_decay_factor 0.7

  @doc """
  Searches for similar content using the given query text.

  Embeds the query (with role `:embedding`, configurable via opts)
  and runs nearest-neighbour matching.

  ## Options
    * `:limit` - Maximum number of results to return (default: 20)
    * `:decay_factor` - Decay factor for multi-chunk scoring (default: 0.7)
    * `:embeddable_types` - List of embeddable types to filter by (optional)
    * any opts accepted by `DefactoAI.Embeddings.Client.embed/2` (e.g.
      `:provider`, `:role`, `:plug`)

  ## Returns
  A list of `%{embeddable_type, embeddable_id, embeddable_title, score, chunks}`
  maps, sorted by score descending.
  """
  def search(query, opts \\ []) when is_binary(query) do
    :telemetry.span([:defacto_ai, :similarity_search], %{}, fn ->
      result =
        with {:ok, query_embedding} <- EmbeddingClient.embed(query, opts) do
          {:ok, search_by_embedding(query_embedding, opts)}
        end

      case result do
        {:ok, hits} -> {hits, %{count: length(hits)}}
        {:error, _} = err -> {err, %{}}
      end
    end)
  end

  @doc """
  Searches for similar content using a pre-computed embedding vector.
  """
  def search_by_embedding(query_embedding, opts \\ []) do
    limit = Keyword.get(opts, :limit, @default_limit)
    decay_factor = Keyword.get(opts, :decay_factor, @default_decay_factor)
    embeddable_types = Keyword.get(opts, :embeddable_types, nil)

    # Get more neighbours than needed to account for grouping
    raw_limit = limit * 5

    neighbors =
      Embedding.nearest_neighbors(query_embedding,
        limit: raw_limit,
        embeddable_types: embeddable_types
      )

    neighbors
    |> Enum.group_by(fn n -> {n.embeddable_type, n.embeddable_id} end)
    |> Enum.map(fn {{type, id}, chunks} ->
      sorted_chunks = Enum.sort_by(chunks, & &1.distance)
      score = calculate_decay_score(sorted_chunks, decay_factor)
      first_chunk = List.first(sorted_chunks)

      %{
        embeddable_type: type,
        embeddable_id: id,
        embeddable_title: first_chunk.embeddable_title,
        score: score,
        chunks:
          Enum.map(sorted_chunks, fn c ->
            %{
              chunk_index: c.chunk_index,
              chunk_text: c.chunk_text,
              distance: c.distance
            }
          end)
      }
    end)
    |> Enum.sort_by(& &1.score, :desc)
    |> Enum.take(limit)
  end

  @doc """
  Calculates a decay-weighted score from a list of chunk distances.

  The score is computed as:
    `sum(similarity_i * decay_factor^i) / sum(decay_factor^i)`

  Where similarity = 1 - distance (cosine distance to similarity conversion)
  and i is the rank (0-indexed) of each chunk sorted by distance.
  """
  def calculate_decay_score(sorted_chunks, decay_factor \\ @default_decay_factor) do
    {weighted_sum, weight_sum} =
      sorted_chunks
      |> Enum.with_index()
      |> Enum.reduce({0.0, 0.0}, fn {chunk, index}, {ws, wt} ->
        # Convert distance to similarity (cosine distance is 0-2, we want 0-1)
        similarity = max(0.0, 1.0 - chunk.distance)
        weight = :math.pow(decay_factor, index)

        {ws + similarity * weight, wt + weight}
      end)

    if weight_sum > 0 do
      weighted_sum / weight_sum
    else
      0.0
    end
  end
end
