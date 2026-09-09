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

  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ToolCall

  @default_timeout 180_000

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

  defp empty_state, do: %{buffer: "", content: "", tool_calls: %{}}

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
      json -> apply_json(state, Jason.decode(json))
    end
  end

  defp apply_line(_, state), do: state

  defp apply_json(state, {:ok, %{"choices" => [%{"delta" => delta} | _]}}) when is_map(delta) do
    state
    |> merge_content(delta["content"])
    |> merge_tool_calls(delta["tool_calls"])
  end

  defp apply_json(state, _), do: state

  defp merge_content(state, content) when is_binary(content),
    do: %{state | content: state.content <> content}

  defp merge_content(state, _), do: state

  defp merge_tool_calls(state, calls) when is_list(calls),
    do: Enum.reduce(calls, state, &merge_tool_call/2)

  defp merge_tool_calls(state, _), do: state

  defp merge_tool_call(call, state) do
    index = call["index"] || 0
    fun = call["function"] || %{}
    existing = Map.get(state.tool_calls, index, %{name: nil, call_id: nil, arguments: ""})

    updated = %{
      name: existing.name || fun["name"],
      call_id: existing.call_id || call["id"],
      arguments: existing.arguments <> (fun["arguments"] || "")
    }

    %{state | tool_calls: Map.put(state.tool_calls, index, updated)}
  end

  # --- Message assembly ---

  defp build_message(state) do
    %Message{
      role: :assistant,
      status: :complete,
      content: if(state.content == "", do: nil, else: state.content),
      tool_calls: build_tool_calls(state.tool_calls)
    }
  end

  defp build_tool_calls(tool_calls) do
    tool_calls
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
