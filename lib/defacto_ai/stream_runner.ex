defmodule DefactoAI.StreamRunner do
  @moduledoc """
  Drop-in replacement for `LangChain.Chains.LLMChain.run/1` for *streaming*
  structured-output requests.

  LangChain's own streaming parser is strict about SSE chunk shape (it expects
  an `index` on every choice, etc.) and silently yields an empty response on
  some OpenAI-compatible gateways — notably Heroku Inference — which then
  surfaces as `"LLM returned an empty response"` and fails every strategy.

  This runner builds the request body with `ChatOpenAI.for_api/3` (so messages,
  tools, `tool_choice` and `response_format` are serialised exactly as
  LangChain would) but sends it through a permissive `Req` SSE collector that
  only reads `choices[0].delta.content` and `choices[0].delta.tool_calls`,
  ignoring anything else — the same tolerant approach `DefactoAI.Client.LangChain`
  already uses for plain `stream_chat/2`.

  It assembles the streamed deltas into a single assistant `%Message{}` (with
  `content` and/or `tool_calls`) and returns `{:ok, chain}` so the strategy's
  `decode_payload/1` and the repair loop work unchanged.
  """

  require Logger

  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ToolCall

  @default_timeout 180_000

  # Diagnostics for unrecognised tool-call chunk shapes: how many raw chunks
  # to keep and how long each may be when logged.
  @raw_delta_sample_size 5
  @raw_delta_max_chars 300

  @doc """
  Run a prepared streaming chain. Mirrors `LLMChain.run/1`'s return shape:
  `{:ok, chain}` on success, `{:error, chain, reason}` on failure.

  `opts` accepts `:plug` for testing (passed straight to `Req`).
  """
  @spec run(LLMChain.t(), keyword()) :: {:ok, LLMChain.t()} | {:error, LLMChain.t(), term()}
  def run(%LLMChain{llm: %ChatOpenAI{} = llm} = chain, opts \\ []) do
    body = ChatOpenAI.for_api(llm, chain.messages, chain.tools)
    timeout = llm.receive_timeout || @default_timeout

    req_opts =
      [
        url: llm.endpoint,
        json: body,
        auth: {:bearer, llm.api_key},
        receive_timeout: timeout,
        retry: false,
        decode_body: false,
        compressed: false,
        into: collector()
      ]
      |> maybe_put(:plug, Keyword.get(opts, :plug))

    case Req.post(req_opts) do
      {:ok, %Req.Response{status: 200} = resp} ->
        message =
          resp
          |> Req.Response.get_private(:defacto_stream, empty_state())
          |> build_message()

        {:ok, %{chain | last_message: message, messages: chain.messages ++ [message]}}

      {:ok, %Req.Response{status: status} = resp} ->
        {:error, chain, {:api_error, status, error_body(resp)}}

      {:error, exception} ->
        {:error, chain, {:http_error, http_error_reason(exception)}}
    end
  end

  # --- SSE collection ---

  # On a 200 the body is an SSE stream and we only care about `data:` lines.
  # On any other status the gateway sends a plain (usually JSON) error body,
  # which has no `data:` lines and would otherwise be swallowed by the SSE
  # parser — so buffer it verbatim and surface it in the `{:api_error, ...}`.
  defp collector do
    fn
      {:data, data}, {req, %Req.Response{status: 200} = resp} ->
        state = Req.Response.get_private(resp, :defacto_stream, empty_state())
        {:cont, {req, Req.Response.put_private(resp, :defacto_stream, consume(state, data))}}

      {:data, data}, {req, resp} ->
        {:cont, {req, append_error_body(resp, data)}}
    end
  end

  defp append_error_body(resp, data) do
    buffered = Req.Response.get_private(resp, :error_body, "")
    Req.Response.put_private(resp, :error_body, buffered <> data)
  end

  defp error_body(%Req.Response{body: body} = resp) do
    case Req.Response.get_private(resp, :error_body, "") do
      "" -> body_to_text(body)
      text -> text
    end
  end

  # `last_started` is the slot of the most recently *started* tool call (one
  # that received an id or a name), so argument deltas without a usable
  # `index` can be attached to it. `raw_tool_deltas` keeps the first few
  # tool-call chunks verbatim for diagnostics only; they never leave
  # `build_message/1`.
  defp empty_state do
    %{
      buffer: "",
      content: "",
      tool_calls: %{},
      last_started: nil,
      raw_tool_deltas: [],
      raw_tool_delta_count: 0,
      # Diagnostics: how many `data:` JSON frames we parsed, how many of them
      # we did not recognise as delta/message chunks, and one such frame.
      frames: 0,
      unrecognised: 0,
      unrecognised_sample: nil
    }
  end

  defp consume(state, data) do
    {lines, buffer} = split_lines(state.buffer <> data)
    Enum.reduce(lines, %{state | buffer: buffer}, &apply_line/2)
  end

  # SSE lines may straddle chunk boundaries — keep a trailing incomplete line
  # in the buffer to be joined with the next chunk's data.
  defp split_lines(data) do
    lines = String.split(data, "\n")

    if String.ends_with?(data, "\n") do
      {lines, ""}
    else
      {Enum.drop(lines, -1), List.last(lines) || ""}
    end
  end

  defp apply_line("data:" <> rest, state) do
    case String.trim_leading(rest, " ") do
      "" -> state
      "[DONE]" -> state
      json -> apply_json(%{state | frames: state.frames + 1}, Jason.decode(json), json)
    end
  end

  defp apply_line(_, state), do: state

  # Standard OpenAI streaming puts partial content in `choices[0].delta`.
  # Some gateways also emit a complete `choices[0].message` (the non-delta
  # form) inside the stream; treat its tool calls as finished. Anything else
  # is counted (and sampled once) for the empty-arguments diagnostics.
  defp apply_json(state, {:ok, %{"choices" => [%{"delta" => delta} | _]}}, _raw)
       when is_map(delta),
       do: merge_choice(state, delta, :delta)

  defp apply_json(state, {:ok, %{"choices" => [%{"message" => message} | _]}}, _raw)
       when is_map(message),
       do: merge_choice(state, message, :complete)

  defp apply_json(state, _decoded, raw) do
    %{
      state
      | unrecognised: state.unrecognised + 1,
        unrecognised_sample: state.unrecognised_sample || truncate(raw, @raw_delta_max_chars)
    }
  end

  defp merge_choice(state, choice, mode) do
    state
    |> merge_content(choice["content"])
    |> merge_tool_calls(choice["tool_calls"], mode)
    |> merge_function_call(choice["function_call"], mode)
  end

  defp merge_content(state, content) when is_binary(content),
    do: %{state | content: state.content <> content}

  defp merge_content(state, _), do: state

  defp merge_tool_calls(state, calls, mode) when is_list(calls),
    do: Enum.reduce(calls, state, &merge_tool_call(&2, &1, mode))

  defp merge_tool_calls(state, _, _mode), do: state

  # Legacy `function_call` shape: a single call without id or index. Treat it
  # as tool call index 0.
  defp merge_function_call(state, %{} = function_call, mode),
    do: merge_tool_call(state, %{"index" => 0, "function" => function_call}, mode)

  defp merge_function_call(state, _, _mode), do: state

  # Tool-call chunks are not uniformly shaped across gateways. OpenAI sends
  # every chunk with an `index`. Anthropic-backed gateways (e.g. Heroku
  # Inference) open the call with `index`/`id`/`function.name` and then stream
  # argument deltas whose `function.name` is `""` and whose `index` is the
  # Anthropic *content-block* index — so a text block preceding the tool_use
  # puts the deltas on a different index than the opening chunk. Keying purely
  # on `index` left the real call with empty arguments. Resolve the slot a
  # chunk belongs to before merging.
  defp merge_tool_call(state, call, mode) when is_map(call) do
    call = normalise_call(call)
    index = target_index(state, call, mode)
    fun = call["function"]
    existing = Map.get(state.tool_calls, index, empty_slot())

    updated = %{
      name: existing.name || fun["name"],
      call_id: existing.call_id || call["id"],
      arguments: merge_arguments(existing.arguments, fun["arguments"], mode)
    }

    %{state | tool_calls: Map.put(state.tool_calls, index, updated)}
    |> adopt_orphans(index, updated)
    |> note_started(index, updated)
    |> remember_raw_delta(call)
  end

  defp merge_tool_call(state, _call, _mode), do: state

  defp empty_slot, do: %{name: nil, call_id: nil, arguments: ""}

  # Blank ids/names (`""` on Heroku's argument deltas) mean "not given"; they
  # must never open a call or overwrite a real name.
  defp normalise_call(call) do
    fun = call["function"] || %{}

    call
    |> Map.put("id", blank_to_nil(call["id"]))
    |> Map.put("function", Map.put(fun, "name", blank_to_nil(fun["name"])))
  end

  defp blank_to_nil(value) when is_binary(value) and value != "", do: value
  defp blank_to_nil(_), do: nil

  defp opens_call?(call), do: call["id"] != nil or call["function"]["name"] != nil

  # A slot is started once a chunk gave it an id or a name. Slots holding
  # only arguments are orphans waiting for their call to open.
  defp started?(%{call_id: call_id, name: name}), do: call_id != nil or name != nil

  defp target_index(state, call, mode) do
    index = call["index"]
    at_index = is_integer(index) && Map.get(state.tool_calls, index)
    last_started = state.last_started && Map.get(state.tool_calls, state.last_started)

    cond do
      # Pure argument delta (no id, no non-empty name). An explicit index
      # pointing at a started call is honoured (standard OpenAI shape);
      # otherwise it continues the most recently started call regardless of
      # its index; before any call opened it parks in an orphan slot.
      not opens_call?(call) ->
        cond do
          at_index && started?(at_index) -> index
          last_started != nil -> state.last_started
          is_integer(index) -> index
          true -> 0
        end

      # The chunk names a call. Landing on an existing slot that is not
      # visibly another call (repeated id/name, or an orphan slot) merges.
      at_index && same_call?(at_index, call) ->
        index

      # A free explicit index opens there.
      is_integer(index) ->
        index

      # No usable index: continue the last started call when this is visibly
      # the same one, otherwise open a new slot.
      last_started != nil and continues?(last_started, call, mode) ->
        state.last_started

      true ->
        next_index(state)
    end
  end

  # A chunk carrying an id different from the slot's id belongs to another call.
  defp same_call?(%{call_id: existing_id}, call) do
    call["id"] == nil or existing_id == nil or call["id"] == existing_id
  end

  defp continues?(slot, call, mode) do
    if call["id"] != nil do
      call["id"] == slot.call_id
    else
      # Gateways may repeat the function name on every delta of one call; a
      # complete (non-delta) message naming a tool is a whole new call.
      mode == :delta and call["function"]["name"] == slot.name
    end
  end

  defp next_index(%{tool_calls: tool_calls}) when map_size(tool_calls) == 0, do: 0
  defp next_index(%{tool_calls: tool_calls}), do: Enum.max(Map.keys(tool_calls)) + 1

  # Deltas concatenate; a complete (non-delta) message carries the whole
  # argument string and replaces any partial text already collected.
  defp merge_arguments(existing, nil, _mode), do: existing
  defp merge_arguments(existing, args, :delta) when is_binary(args), do: existing <> args
  defp merge_arguments(existing, "", :complete), do: existing
  defp merge_arguments(_existing, args, :complete) when is_binary(args), do: args
  defp merge_arguments(existing, _args, _mode), do: existing

  # Argument deltas that arrived before their call opened sit in orphan
  # slots. Once the first — and only — started call exists, fold them into
  # it in index order.
  defp adopt_orphans(state, index, slot) do
    {orphans, started} =
      Enum.split_with(state.tool_calls, fn {_index, s} -> not started?(s) end)

    if started?(slot) and orphans != [] and length(started) == 1 do
      prefix =
        orphans
        |> Enum.sort_by(fn {orphan_index, _} -> orphan_index end)
        |> Enum.map_join("", fn {_index, s} -> s.arguments end)

      %{state | tool_calls: %{index => %{slot | arguments: prefix <> slot.arguments}}}
    else
      state
    end
  end

  defp note_started(state, index, slot) do
    if started?(slot), do: %{state | last_started: index}, else: state
  end

  defp remember_raw_delta(state, call) do
    deltas =
      if length(state.raw_tool_deltas) < @raw_delta_sample_size,
        do: state.raw_tool_deltas ++ [call],
        else: state.raw_tool_deltas

    %{state | raw_tool_deltas: deltas, raw_tool_delta_count: state.raw_tool_delta_count + 1}
  end

  # --- Message assembly ---

  defp build_message(state) do
    maybe_log_empty_arguments(state)

    %Message{
      role: :assistant,
      status: :complete,
      content: if(state.content == "", do: nil, else: state.content),
      tool_calls: build_tool_calls(state.tool_calls)
    }
  end

  # A tool call that finished with no arguments means we did not receive (or
  # did not recognise) the gateway's argument deltas. Log a bounded picture of
  # what *did* arrive — raw tool-call chunks, assembled content, frame counts
  # and one unrecognised frame — so the real shape shows up in production
  # logs instead of just "validation failed".
  defp maybe_log_empty_arguments(%{tool_calls: tool_calls} = state)
       when map_size(tool_calls) > 0 do
    if Enum.any?(tool_calls, fn {_index, call} -> call.arguments == "" end) do
      sample =
        state.raw_tool_deltas
        |> Enum.map(&truncate(inspect(&1), @raw_delta_max_chars))
        |> Enum.join(" | ")

      Logger.info(fn ->
        "DefactoAI.StreamRunner: streamed tool call finished with empty arguments; " <>
          "content_length=#{String.length(state.content)} " <>
          "frames=#{state.frames} unrecognised=#{state.unrecognised} " <>
          "#{state.raw_tool_delta_count} raw tool-call chunk(s) seen, sample: #{sample}; " <>
          "content_sample: #{inspect(truncate(state.content, @raw_delta_max_chars))}; " <>
          "unrecognised_sample: #{inspect(state.unrecognised_sample)}"
      end)
    end

    :ok
  end

  defp maybe_log_empty_arguments(_state), do: :ok

  defp truncate(text, max) when byte_size(text) <= max, do: text
  defp truncate(text, max), do: String.slice(text, 0, max - 1) <> "…"

  # Orphan slots (arguments without id or name) are only meaningful when no
  # call ever opened; once one did, whatever they hold was either adopted or
  # is noise, so they are dropped.
  defp build_tool_calls(tool_calls) do
    slots = Map.to_list(tool_calls)

    slots =
      if Enum.any?(slots, fn {_index, slot} -> started?(slot) end),
        do: Enum.filter(slots, fn {_index, slot} -> started?(slot) end),
        else: slots

    slots
    |> Enum.sort_by(fn {index, _} -> index end)
    |> Enum.map(fn {index, tc} ->
      %ToolCall{
        type: :function,
        status: :complete,
        index: index,
        name: tc.name,
        call_id: tc.call_id,
        arguments: parse_arguments(tc.arguments)
      }
    end)
  end

  defp parse_arguments(""), do: %{}

  defp parse_arguments(arguments) when is_binary(arguments) do
    case Jason.decode(arguments) do
      {:ok, map} when is_map(map) -> map
      _ -> arguments
    end
  end

  # --- Helpers ---

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)

  defp body_to_text(body) when is_binary(body), do: body
  defp body_to_text(body), do: inspect(body)

  defp http_error_reason(%{reason: reason}), do: reason
  defp http_error_reason(other), do: other
end
