defmodule DefactoAI.JSON do
  @moduledoc """
  Pure-function pipeline for turning LLM output into validated structs.

  Handles the messy reality that even when a model is asked for "JSON and
  only JSON", it sometimes wraps the payload in code fences, prefixes it
  with prose, or emits common syntactic mistakes (trailing commas, smart
  quotes). The pipeline is:

      raw text  --extract--> term  --cast--> struct
                      \\
                       --(on failure)--> repair --> term --> cast --> struct

  No LLM calls happen here — repair is purely textual. The corrective
  retry-with-error loop lives in `DefactoAI.RepairLoop`.
  """

  @type schema_module :: module()
  @type cast_opts :: keyword()

  @doc """
  Pulls a JSON value out of arbitrary text and decodes it.

  Strips ```json fences and surrounding prose, then slices from the first
  `{` or `[` to its matching close (string-aware bracket counting), and
  decodes with Jason.
  """
  @spec extract(binary()) :: {:ok, term()} | {:error, :no_json | {:invalid_json, term()}}
  def extract(text) when is_binary(text) do
    text
    |> strip_code_fences()
    |> slice_balanced()
    |> case do
      :no_json -> {:error, :no_json}
      json -> decode(json)
    end
  end

  @doc """
  Best-effort textual repair for common LLM JSON mistakes, then decode.

  Fixes trailing commas before `}`/`]`, normalises smart quotes to ASCII
  double quotes, and re-runs the slice/decode. Does not attempt to repair
  unterminated strings or invented keys — those should bounce out to the
  retry-with-error loop.
  """
  @spec repair(binary()) :: {:ok, term()} | {:error, :no_json | {:invalid_json, term()}}
  def repair(text) when is_binary(text) do
    text
    |> normalise_smart_quotes()
    |> strip_trailing_commas()
    |> extract()
  end

  @doc """
  Casts a decoded map into the given Ecto-embedded schema and runs its
  changeset.

  The schema module must export `changeset/2` or `changeset/3` (the latter
  receives `opts` so schemas with a validation context can use it). Returns
  `{:ok, struct}` or `{:error, Ecto.Changeset.t()}`.
  """
  @spec cast(map() | list(), schema_module(), cast_opts()) ::
          {:ok, struct()} | {:error, Ecto.Changeset.t()}
  def cast(data, schema_module, opts \\ []) when is_atom(schema_module) do
    Code.ensure_loaded(schema_module)

    changeset =
      if function_exported?(schema_module, :changeset, 3) do
        schema_module.changeset(struct(schema_module), data, opts)
      else
        schema_module.changeset(struct(schema_module), data)
      end

    Ecto.Changeset.apply_action(changeset, :insert)
  end

  @doc """
  Full pipeline: extract → (repair on failure) → cast.

  When given a map (e.g. the parsed arguments of a tool call) the extract
  and repair stages are skipped.
  """
  @spec decode_and_cast(binary() | map(), schema_module(), cast_opts()) ::
          {:ok, struct()} | {:error, term()}
  def decode_and_cast(data, schema_module, opts \\ [])

  def decode_and_cast(data, schema_module, opts) when is_map(data) do
    cast(data, schema_module, opts)
  end

  def decode_and_cast(text, schema_module, opts) when is_binary(text) do
    case extract(text) do
      {:ok, decoded} ->
        cast(decoded, schema_module, opts)

      {:error, _} = first_error ->
        case repair(text) do
          {:ok, decoded} -> cast(decoded, schema_module, opts)
          {:error, _} -> first_error
        end
    end
  end

  # ---------------------------------------------------------------------------
  # Internals
  # ---------------------------------------------------------------------------

  defp decode(text) do
    case Jason.decode(text) do
      {:ok, value} -> {:ok, value}
      {:error, reason} -> {:error, {:invalid_json, reason}}
    end
  end

  defp strip_code_fences(text) do
    text
    |> String.trim()
    |> String.replace(~r/^```(?:json)?\s*\n?/i, "")
    |> String.replace(~r/\n?```\s*$/, "")
    |> String.trim()
  end

  # Walk the binary looking for the first `{` or `[`, then track depth
  # (string-aware) until the matching close. Returns the sliced JSON or
  # `:no_json` if no balanced span is found.
  defp slice_balanced(text) do
    case find_open(text, 0) do
      nil -> :no_json
      start_index -> scan_close(text, start_index)
    end
  end

  defp find_open(text, index) do
    case :binary.match(text, ["{", "["], scope: {index, byte_size(text) - index}) do
      {pos, _len} -> pos
      :nomatch -> nil
    end
  end

  defp scan_close(text, start_index) do
    open_char = :binary.at(text, start_index)
    close_char = matching_close(open_char)
    do_scan(text, start_index + 1, 1, false, false, close_char, open_char, start_index)
  end

  defp matching_close(?{), do: ?}
  defp matching_close(?[), do: ?]

  # Returns the sliced JSON binary when balance reaches 0, or :no_json.
  defp do_scan(text, index, _depth, _in_string, _escape, _close, _open, _start)
       when index >= byte_size(text),
       do: :no_json

  defp do_scan(text, index, depth, in_string, escape, close, open, start) do
    char = :binary.at(text, index)

    cond do
      escape ->
        do_scan(text, index + 1, depth, in_string, false, close, open, start)

      in_string and char == ?\\ ->
        do_scan(text, index + 1, depth, true, true, close, open, start)

      in_string and char == ?" ->
        do_scan(text, index + 1, depth, false, false, close, open, start)

      in_string ->
        do_scan(text, index + 1, depth, true, false, close, open, start)

      char == ?" ->
        do_scan(text, index + 1, depth, true, false, close, open, start)

      char == open ->
        do_scan(text, index + 1, depth + 1, false, false, close, open, start)

      char == close and depth == 1 ->
        :binary.part(text, start, index - start + 1)

      char == close ->
        do_scan(text, index + 1, depth - 1, false, false, close, open, start)

      true ->
        do_scan(text, index + 1, depth, false, false, close, open, start)
    end
  end

  # ---- Repair helpers -------------------------------------------------------

  defp normalise_smart_quotes(text) do
    text
    |> String.replace(["“", "”", "„", "‟"], "\"")
    |> String.replace(["‘", "’", "‚", "‛"], "'")
  end

  defp strip_trailing_commas(text) do
    String.replace(text, ~r/,(\s*[}\]])/, "\\1")
  end
end
