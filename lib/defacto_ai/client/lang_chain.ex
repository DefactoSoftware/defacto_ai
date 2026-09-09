defmodule DefactoAI.Client.LangChain do
  @moduledoc """
  Production implementation of `DefactoAI.Client`.

  Resolves a provider, walks the configured strategy chain
  (`DefactoAI.Strategy.default_order/0` by default), and runs the
  inner repair-and-retry loop (`DefactoAI.RepairLoop`) inside each
  strategy. Falls back to the next strategy when a provider rejects
  the current one, or when the strategy returns a response that
  can't be decoded into a payload.

  Validation failures (the cast through the response schema's
  changeset returns an error) trigger the repair loop within the
  *same* strategy — they do not trigger fallback. Falling back on a
  validation failure would mask prompt bugs.
  """

  @behaviour DefactoAI.Client

  require Logger

  alias DefactoAI.LangChainAdapter
  alias DefactoAI.Provider
  alias DefactoAI.RepairLoop
  alias DefactoAI.Strategy
  alias LangChain.Chains.LLMChain
  alias LangChain.LangChainError
  alias LangChain.Message

  @default_max_validation_retries 2
  @default_stream_timeout 180_000

  @impl true
  def complete_structured(schema_module, messages, opts \\ []) do
    with {:ok, provider} <- resolve_provider(opts) do
      strategies = Keyword.get(opts, :strategies, configured_strategies())
      budget = Keyword.get(opts, :max_validation_retries, @default_max_validation_retries)
      validation_opts = validation_opts(opts)

      try_strategies(strategies, provider, schema_module, messages, budget, validation_opts, [])
    end
  end

  # Strategy order can be overridden globally via the :default_strategies
  # Application env. Useful when a specific provider misbehaves on the
  # default first-choice strategy (e.g. Heroku Inference's long tool-call
  # latency pushing requests over the 30s gateway timeout — operators can
  # set [JsonMode, TextRepair] to skip ToolCall entirely).
  defp configured_strategies do
    Application.get_env(:defacto_ai, :default_strategies, Strategy.default_order())
  end

  defp try_strategies([], _provider, _schema, _messages, _budget, _opts, errors) do
    {:error, {:all_strategies_failed, Enum.reverse(errors)}}
  end

  defp try_strategies([strategy | rest], provider, schema, messages, budget, opts, errors) do
    case strategy.prepare(provider, schema, messages, opts) do
      {:ok, chain} ->
        case RepairLoop.run(chain, schema, &strategy.decode_payload/1, budget, opts) do
          {:ok, struct} ->
            {:ok, struct}

          {:error, {kind, _} = reason} when kind in [:strategy_unsupported, :decode_failed] ->
            log_fallback(strategy, reason)

            try_strategies(
              rest,
              provider,
              schema,
              messages,
              budget,
              opts,
              [{strategy, reason} | errors]
            )

          {:error, _other} = error ->
            error
        end

      {:error, :unsupported} = error ->
        log_fallback(strategy, error)

        try_strategies(rest, provider, schema, messages, budget, opts, [{strategy, error} | errors])

      {:error, _} = error ->
        error
    end
  end

  @doc false
  # Public for use by other modules in the library (e.g. the embeddings
  # client and similarity search). Hosts should not call this directly.
  @spec resolve_provider(keyword()) :: {:ok, term()} | {:error, term()}
  def resolve_provider(opts) do
    case Keyword.get(opts, :provider) do
      nil ->
        role = Keyword.get(opts, :role, :llm)
        resolve_by_role(role)

      provider ->
        {:ok, provider}
    end
  end

  defp resolve_by_role(role) do
    case Application.get_env(:defacto_ai, :provider_resolver) do
      nil ->
        {:error, :no_provider_resolver}

      fun when is_function(fun, 1) ->
        case fun.(role) do
          nil -> {:error, {:no_provider_for_role, role}}
          provider -> {:ok, provider}
        end
    end
  end

  # Build the keyword list that gets handed to the response schema's
  # changeset/3. The full caller opts pass through (schemas ignore keys
  # they don't recognise), but if `:validation_context` is set we lift
  # its entries into the top level so schemas can read them as plain
  # keyword opts (e.g. `opts[:valid_topic_ids]`) instead of having to
  # reach through a nested map.
  defp validation_opts(opts) do
    case Keyword.get(opts, :validation_context) do
      nil -> opts
      ctx when is_map(ctx) -> Keyword.merge(opts, Enum.to_list(ctx))
      ctx when is_list(ctx) -> Keyword.merge(opts, ctx)
    end
  end

  defp log_fallback(strategy, reason) do
    Logger.info(fn ->
      "DefactoAI: strategy #{inspect(strategy)} unavailable, falling back. Reason: #{inspect(reason)}"
    end)

    :telemetry.execute(
      [:defacto_ai, :strategy, :rejected],
      %{count: 1},
      %{strategy: strategy, reason: reason}
    )
  end

  # ---------------------------------------------------------------------------
  # Plain chat (sync + streaming)
  # ---------------------------------------------------------------------------

  @impl true
  def complete_chat(messages, opts \\ []) do
    with {:ok, provider} <- resolve_provider(opts) do
      chat_model = LangChainAdapter.build_chat_model(provider, [stream: false], opts)

      chat_model
      |> LangChainAdapter.build_chain(messages)
      |> LLMChain.run()
      |> handle_chat_run_result()
    end
  end

  @impl true
  def stream_chat(messages, opts \\ []) do
    with {:ok, provider} <- resolve_provider(opts) do
      do_stream_chat(provider, messages, opts)
    end
  end

  # Streaming uses Req directly (not LangChain) with a deliberately
  # permissive SSE parser that just extracts `choices[0].delta.content`
  # and drops anything else. LangChain's stream parser is strict about
  # chunk shape (e.g. requires `index` on every choice) and silently
  # converts unrecognised chunks into errors, which the OpenAI-compat
  # gateway some operators use does not satisfy.
  defp do_stream_chat(provider, messages, opts) do
    caller = self()
    ref = make_ref()
    timeout = Keyword.get(opts, :stream_timeout, @default_stream_timeout)

    task =
      Task.async(fn ->
        run_stream_request(provider, messages, opts, caller, ref, timeout)
      end)

    stream =
      Stream.resource(
        fn -> %{ref: ref, task: task, timeout: timeout, halt: false} end,
        fn
          %{halt: true} = state ->
            {:halt, state}

          %{ref: ref, timeout: timeout} = state ->
            receive do
              {^ref, {:chunk, content}} ->
                {[content], state}

              {^ref, :done} ->
                {:halt, state}

              {^ref, {:error, reason}} ->
                # Emit the error and ensure the next pull halts the stream
                # rather than blocking on another receive.
                {[{:error, reason}], %{state | halt: true}}
            after
              timeout ->
                {[{:error, :timeout}], %{state | halt: true}}
            end
        end,
        fn %{task: task} -> Task.shutdown(task, :brutal_kill) end
      )

    {:ok, stream}
  end

  defp run_stream_request(provider, messages, opts, caller, ref, timeout) do
    url = chat_endpoint(provider)

    body =
      %{
        model: Provider.model(provider),
        messages: Enum.map(messages, &normalise_message/1),
        stream: true
      }
      |> Map.merge(stream_body_attrs(opts))

    req_opts =
      [
        url: url,
        json: body,
        auth: {:bearer, Provider.api_key(provider)},
        receive_timeout: timeout,
        retry: false,
        decode_body: false,
        compressed: false,
        into: stream_collector(caller, ref)
      ]
      |> maybe_put(:plug, Keyword.get(opts, :plug))

    case Req.post(req_opts) do
      {:ok, %Req.Response{status: 200}} ->
        send(caller, {ref, :done})

      {:ok, %Req.Response{status: status} = resp} ->
        send(caller, {ref, {:error, {:api_error, status, error_body(resp)}}})

      {:error, exception} ->
        send(caller, {ref, {:error, {:http_error, http_error_reason(exception)}}})
    end
  end

  # Req's `into:` callback is invoked once per HTTP body chunk, with
  # `{req, resp}` threaded through. We buffer any incomplete trailing
  # SSE line on the response's private dict so it can be re-joined with
  # the next chunk's data — SSE lines may straddle chunk boundaries.
  #
  # Non-200 responses carry a plain (usually JSON) error body rather than
  # SSE, which the `data:`-line parser would silently discard. Buffer those
  # bytes verbatim instead so the provider's message reaches the caller.
  defp stream_collector(caller, ref) do
    fn
      {:data, data}, {req, %Req.Response{status: 200} = resp} ->
        buffer = Req.Response.get_private(resp, :sse_buffer, "")
        {chunks, new_buffer} = parse_sse_chunks(buffer <> data)

        for content <- chunks do
          send(caller, {ref, {:chunk, content}})
        end

        {:cont, {req, Req.Response.put_private(resp, :sse_buffer, new_buffer)}}

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

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)

  # The streaming request is built by hand rather than through ChatOpenAI,
  # so only pass through the per-call chat model attrs that map 1:1 onto
  # request-body fields.
  defp stream_body_attrs(opts) do
    opts
    |> LangChainAdapter.chat_model_attrs()
    |> Map.take([:max_tokens, :temperature])
  end

  defp chat_endpoint(provider) do
    base = Provider.base_url(provider) || ""
    path = Provider.api_path(provider) || ""
    String.trim_trailing(base, "/") <> path
  end

  defp normalise_message(%LangChain.Message{role: role, content: content}) do
    %{role: role_to_string(role), content: content_to_string(content)}
  end

  defp normalise_message(%{role: role, content: content}) do
    %{role: role_to_string(role), content: content_to_string(content)}
  end

  defp normalise_message(%{"role" => role, "content" => content}) do
    %{role: role_to_string(role), content: content_to_string(content)}
  end

  defp role_to_string(role) when is_binary(role), do: role
  defp role_to_string(role) when is_atom(role), do: Atom.to_string(role)

  defp content_to_string(content) when is_binary(content), do: content

  defp content_to_string(parts) when is_list(parts) do
    Enum.map_join(parts, "", fn
      %LangChain.Message.ContentPart{type: :text, content: c} when is_binary(c) -> c
      _ -> ""
    end)
  end

  defp parse_sse_chunks(data) do
    lines = String.split(data, "\n")

    {complete_lines, buffer} =
      if String.ends_with?(data, "\n") do
        {lines, ""}
      else
        {Enum.drop(lines, -1), List.last(lines) || ""}
      end

    chunks = Enum.flat_map(complete_lines, &parse_sse_line/1)
    {chunks, buffer}
  end

  defp parse_sse_line("data:" <> rest) do
    case String.trim_leading(rest, " ") do
      "" -> []
      "[DONE]" -> []
      json -> parse_json_chunk(json)
    end
  end

  defp parse_sse_line(_), do: []

  defp parse_json_chunk(json) do
    case Jason.decode(json) do
      {:ok, %{"choices" => [%{"delta" => %{"content" => content}} | _]}}
      when is_binary(content) and content != "" ->
        [content]

      _ ->
        []
    end
  end

  defp body_to_text(body) when is_binary(body), do: body
  defp body_to_text(body), do: inspect(body)

  defp http_error_reason(%Req.TransportError{reason: reason}), do: reason
  defp http_error_reason(%Mint.TransportError{reason: reason}), do: reason
  defp http_error_reason(%{reason: reason}), do: reason
  defp http_error_reason(other), do: other

  defp handle_chat_run_result({:ok, %LLMChain{} = chain}), do: extract_content(chain)
  defp handle_chat_run_result({:ok, %LLMChain{} = chain, _last}), do: extract_content(chain)

  defp handle_chat_run_result({:error, _chain, %LangChainError{} = err}),
    do: {:error, classify_chat_error(err)}

  defp handle_chat_run_result({:error, _chain, other}), do: {:error, other}

  defp extract_content(%LLMChain{last_message: %Message{content: content}})
       when is_binary(content),
       do: {:ok, content}

  defp extract_content(%LLMChain{last_message: %Message{content: parts}}) when is_list(parts) do
    text =
      parts
      |> Enum.map(fn
        %LangChain.Message.ContentPart{type: :text, content: c} when is_binary(c) -> c
        _ -> ""
      end)
      |> Enum.join()

    if text == "", do: {:error, :unexpected_response}, else: {:ok, text}
  end

  defp extract_content(_), do: {:error, :unexpected_response}

  # Map LangChain errors back into the {:api_error, status, body} /
  # {:http_error, reason} / :timeout / :unexpected_response shapes the
  # existing RAG error logging already understands.
  defp classify_chat_error(%LangChainError{type: "timeout"}), do: :timeout

  defp classify_chat_error(%LangChainError{type: "empty_response"}), do: :unexpected_response

  defp classify_chat_error(%LangChainError{original: %{"error" => %{"code" => code}}})
       when is_integer(code),
       do: {:api_error, code}

  defp classify_chat_error(%LangChainError{message: msg}) when is_binary(msg) do
    cond do
      msg =~ ~r/timeout|timed\s*out/i -> :timeout
      msg =~ ~r/connection|unreachable|closed|refused/i -> {:http_error, msg}
      true -> {:api_error, msg}
    end
  end

  defp classify_chat_error(other), do: other
end
