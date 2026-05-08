defmodule DefactoAI.Embeddings.Chunker do
  @moduledoc """
  Splits text content into overlapping chunks for embedding.

  Uses a sliding window approach with word-boundary aware splitting.
  """

  # API limit is 2048 chars, use 1200 to leave buffer for encoding overhead
  @default_chunk_size 1200
  @default_overlap 120

  @doc """
  Splits text into overlapping chunks.

  ## Options
    * `:chunk_size` - Target size for each chunk in characters (default: 1200)
    * `:overlap` - Number of characters to overlap between chunks (default: 120)

  ## Returns
  A list of maps with:
    * `:content` - The chunk text
    * `:index` - Zero-based index of the chunk
    * `:start_position` - Character position where chunk starts in original text
    * `:end_position` - Character position where chunk ends in original text
  """
  def chunk(text, opts \\ [])
  def chunk(nil, _opts), do: []
  def chunk("", _opts), do: []

  def chunk(text, opts) when is_binary(text) do
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    overlap = Keyword.get(opts, :overlap, @default_overlap)

    text = String.trim(text)

    if text == "" do
      []
    else
      if String.length(text) <= chunk_size do
        [%{content: text, index: 0, start_position: 0, end_position: String.length(text)}]
      else
        do_chunk(text, chunk_size, overlap, 0, 0, [])
      end
    end
  end

  defp do_chunk(text, chunk_size, overlap, start_pos, index, acc) do
    remaining = String.slice(text, start_pos..-1//1)
    remaining_length = String.length(remaining)

    cond do
      remaining_length == 0 ->
        Enum.reverse(acc)

      remaining_length <= chunk_size ->
        chunk = %{
          content: remaining,
          index: index,
          start_position: start_pos,
          end_position: start_pos + remaining_length
        }

        Enum.reverse([chunk | acc])

      true ->
        # Get a chunk of the target size
        raw_chunk = String.slice(remaining, 0, chunk_size)

        # Find the last word boundary (space, newline, etc.)
        chunk_content = find_word_boundary(raw_chunk)
        chunk_length = String.length(chunk_content)

        chunk = %{
          content: chunk_content,
          index: index,
          start_position: start_pos,
          end_position: start_pos + chunk_length
        }

        # Move forward by chunk_length minus overlap
        next_start = start_pos + max(chunk_length - overlap, 1)

        do_chunk(text, chunk_size, overlap, next_start, index + 1, [chunk | acc])
    end
  end

  defp find_word_boundary(text) do
    # Try to find the last space or newline to break on
    case String.last(text) do
      " " ->
        String.trim_trailing(text)

      "\n" ->
        String.trim_trailing(text)

      _ ->
        # Find the last word boundary
        case Regex.run(~r/^(.+)[\s\n][^\s\n]*$/s, text) do
          [_, match] -> String.trim_trailing(match)
          nil -> text
        end
    end
  end
end
