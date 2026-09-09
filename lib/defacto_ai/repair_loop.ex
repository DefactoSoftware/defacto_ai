defmodule DefactoAI.RepairLoop do
  @moduledoc """
  Inner loop for a single strategy: run the chain, decode the payload,
  cast it through the schema. On validation failure, append a corrective
  user message describing what went wrong and retry until the budget is
  exhausted.

  This is the *intra-strategy* loop. The *inter-strategy* fallback (when
  a provider rejects tool calling outright, etc.) lives in
  `DefactoAI.Client.LangChain`.
  """

  require Logger

  alias DefactoAI.JSON
  alias LangChain.Chains.LLMChain
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult

  @type decode_fun :: (LLMChain.t() -> {:ok, map() | binary()} | {:error, term()})
  @type result :: {:ok, struct()} | {:error, term()}

  # Stand-in text for an assistant turn whose payload lives entirely in
  # `tool_calls`. See `fill_blank_content/1`.
  @tool_call_placeholder "(structured response provided as a tool call)"

  # Upper bound for the retry reason we log and attach to telemetry.
  @max_reason_length 500

  @doc """
  Run a prepared chain, decode and cast its payload, retrying with a
  corrective message on validation failure.

    * `chain` — a prepared `%LLMChain{}` from a strategy's `prepare/4`.
    * `schema_module` — the response schema module.
    * `decode_fun` — the strategy's `decode_payload/1` function.
    * `budget` — number of corrective retries allowed (0 means no retry).
    * `opts` — passed through to `JSON.decode_and_cast/3` (e.g. for
      `:validation_context`-style schemas that need extra args via
      `changeset/3`).

  Return values:
    * `{:ok, struct}`
    * `{:error, {:validation_failed, term}}` — budget exhausted on a
      cast/parse error.
    * `{:error, {:transient, term}}` — HTTP 5xx, rate-limit, or network
      error from the provider. Caller can retry transparently.
    * `{:error, {:strategy_unsupported, binary}}` — provider rejected
      this strategy (e.g. "tool_choice not supported"). Caller should
      fall back to the next strategy.
    * `{:error, {:decode_failed, term}}` — strategy got a response but
      couldn't extract a payload (e.g. JSON-mode chain returned no
      content). Caller should fall back.
    * `{:error, term}` — anything else.
  """
  @spec run(LLMChain.t(), module(), decode_fun(), non_neg_integer(), keyword()) :: result()
  def run(%LLMChain{} = chain, schema_module, decode_fun, budget, opts \\ [])
      when is_atom(schema_module) and is_function(decode_fun, 1) do
    case run_chain(ensure_message_content(chain), opts) do
      {:ok, new_chain} ->
        decode_and_validate(new_chain, schema_module, decode_fun, budget, opts)

      {:ok, new_chain, _msg} ->
        decode_and_validate(new_chain, schema_module, decode_fun, budget, opts)

      {:error, _chain, %LangChainError{} = err} ->
        classify_chain_error(err)

      {:error, _chain, other} ->
        {:error, other}
    end
  end

  # Streaming requests go through DefactoAI.StreamRunner, whose tolerant SSE
  # parser handles gateways (e.g. Heroku Inference) that LangChain's strict
  # streaming parser chokes on (yielding an empty response). Non-streaming
  # requests keep using LangChain's own runner.
  defp run_chain(%LLMChain{llm: %{stream: true}} = chain, opts) do
    DefactoAI.StreamRunner.run(chain, Keyword.take(opts, [:plug]))
  end

  defp run_chain(%LLMChain{} = chain, _opts), do: LLMChain.run(chain)

  # A forced tool call comes back as an assistant message with blank content
  # (`nil`, `""` or `[]` — the payload lives in `tool_calls`). When the repair
  # loop re-sends that turn on a corrective retry, LangChain serialises the
  # content as-is. OpenAI accepts `null`/`""` alongside `tool_calls`, but
  # gateways fronting Anthropic (e.g. Heroku Inference) require every message
  # to carry non-empty text and reject the history with a 400 such as
  # `messages[2]: content is required` — for `null` *and* for `""`. Substitute
  # a short placeholder so the turn round-trips on those providers without
  # losing the tool calls or our retry budget. Messages without tool calls
  # are left untouched: blank content there is a real signal, not an
  # artefact of tool calling.
  defp ensure_message_content(%LLMChain{messages: messages} = chain) do
    %{chain | messages: Enum.map(messages, &fill_blank_content/1)}
  end

  defp fill_blank_content(
         %Message{role: :assistant, content: content, tool_calls: [_ | _]} = message
       )
       when content in [nil, "", []] do
    %{message | content: @tool_call_placeholder}
  end

  defp fill_blank_content(%Message{} = message), do: message

  defp decode_and_validate(chain, schema_module, decode_fun, budget, opts) do
    case decode_fun.(chain) do
      {:ok, payload} ->
        case JSON.decode_and_cast(payload, schema_module, opts) do
          {:ok, struct} ->
            {:ok, struct}

          {:error, validation_error} when budget > 0 ->
            reason = retry_reason(schema_module, decode_fun, validation_error, payload)

            # Info, not debug: in production this is the only place that says
            # *why* the model's first answer was unusable.
            Logger.info(fn ->
              "DefactoAI: validation failed, retrying with corrective message " <>
                "(#{budget - 1} attempts left): #{reason}"
            end)

            :telemetry.execute(
              [:defacto_ai, :repair_loop, :retry],
              %{remaining: budget - 1},
              %{schema: schema_module, reason: reason}
            )

            chain
            |> add_corrective_turn(validation_error, payload)
            |> run(schema_module, decode_fun, budget - 1, opts)

          {:error, validation_error} ->
            {:error, {:validation_failed, validation_error}}
        end

      {:error, decode_error} ->
        {:error, {:decode_failed, decode_error}}
    end
  end

  @doc false
  # The messages appended to the chain before a retry. Public (but hidden)
  # so the exact shape can be asserted in tests; hosts should not call it.
  #
  # Both OpenAI and Anthropic (behind OpenAI-compatible gateways such as
  # Heroku Inference) require that an assistant message carrying tool calls
  # is immediately followed by one tool result per call; Anthropic rejects
  # the history otherwise with `tool_use ids were found without tool_result
  # blocks immediately after`. So when the rejected answer was a tool call,
  # answer every call with an error tool result first, then append the
  # corrective user message. Without tool calls only the user message is sent.
  @spec corrective_messages(LLMChain.t(), term(), term()) :: [Message.t()]
  def corrective_messages(%LLMChain{last_message: last}, error, payload) do
    tool_results(last, error) ++ [corrective_message(error, payload)]
  end

  defp add_corrective_turn(chain, error, payload) do
    chain
    |> corrective_messages(error, payload)
    |> Enum.reduce(chain, &LLMChain.add_message(&2, &1))
  end

  defp tool_results(%Message{role: :assistant, tool_calls: [_ | _] = calls}, error) do
    # A call without an id cannot be answered (and would be rejected by the
    # provider on its own anyway), so it is skipped rather than raising here.
    results =
      for %ToolCall{call_id: call_id, name: name} <- calls, is_binary(call_id) do
        ToolResult.new!(%{
          tool_call_id: call_id,
          name: name,
          content:
            "Rejected: #{describe_error(error)}. " <>
              "A corrected response is requested in the next message.",
          is_error: true
        })
      end

    case results do
      [] -> []
      results -> [Message.new_tool_result!(%{tool_results: results})]
    end
  end

  defp tool_results(_last, _error), do: []

  defp corrective_message(error, payload) do
    Message.new_user!("""
    The previous response could not be used because: #{describe_error(error)}.

    The previous response was:
    #{format_payload(payload)}

    Please respond again. The response must conform to the same schema you
    were given. Return only the JSON value — no prose, no code fences.
    """)
  end

  defp describe_error(%Ecto.Changeset{} = changeset) do
    errors =
      changeset
      |> Ecto.Changeset.traverse_errors(fn {msg, opts} ->
        Enum.reduce(opts, msg, fn {k, v}, acc ->
          String.replace(acc, "%{#{k}}", to_string(v))
        end)
      end)
      |> format_changeset_errors()

    "validation failed (#{errors})"
  end

  defp describe_error({:validation_failed, inner}), do: describe_error(inner)

  defp describe_error({:invalid_json, %Jason.DecodeError{} = err}),
    do: "invalid JSON (#{Exception.message(err)})"

  defp describe_error(:no_json), do: "the response did not contain any JSON"
  defp describe_error(other), do: "unexpected error (#{inspect(other)})"

  defp format_changeset_errors(map) when is_map(map) do
    map
    |> Enum.map(fn
      {field, msgs} when is_list(msgs) -> "#{field}: #{Enum.join(msgs, ", ")}"
      {field, sub} when is_map(sub) -> "#{field} → #{format_changeset_errors(sub)}"
    end)
    |> Enum.join("; ")
  end

  defp format_payload(payload) when is_binary(payload), do: payload
  defp format_payload(payload), do: inspect(payload)

  # Bounded, log-safe description of a validation failure: which strategy and
  # schema, what the changeset complained about, and the *shape* of the
  # payload (top-level keys only — never the full model output).
  defp retry_reason(schema_module, decode_fun, error, payload) do
    text =
      "strategy=#{inspect(strategy_module(decode_fun))} schema=#{inspect(schema_module)} " <>
        "#{describe_error(error)} payload_keys=#{describe_payload_keys(payload)}"

    truncate(text, @max_reason_length)
  end

  defp strategy_module(decode_fun) do
    case Function.info(decode_fun, :module) do
      {:module, module} -> module
      _ -> :unknown
    end
  end

  defp describe_payload_keys(payload) when is_map(payload) do
    keys =
      payload
      |> Map.keys()
      |> Enum.map(fn
        key when is_binary(key) -> key
        key -> inspect(key)
      end)
      |> Enum.sort()
      |> Enum.join(", ")

    "[" <> keys <> "]"
  end

  defp describe_payload_keys(payload) when is_binary(payload),
    do: "<#{byte_size(payload)}-byte text>"

  defp describe_payload_keys(_payload), do: "<non-map>"

  defp truncate(text, max) when byte_size(text) <= max, do: text
  defp truncate(text, max), do: String.slice(text, 0, max - 1) <> "…"

  defp classify_chain_error(%LangChainError{type: type, message: msg} = err) do
    cond do
      # LangChain raises this when a streamed response produces no
      # messages or deltas. Common with providers that can stream
      # plain content but not tool-call deltas — fall back to a
      # strategy that doesn't rely on tool calling.
      type == "empty_response" ->
        {:error, {:decode_failed, msg}}

      is_binary(msg) and strategy_unsupported?(msg) ->
        {:error, {:strategy_unsupported, msg}}

      is_binary(msg) and transient?(msg) ->
        {:error, {:transient, msg}}

      true ->
        {:error, err}
    end
  end

  defp strategy_unsupported?(msg) do
    cond do
      # Direct "thing X not supported" — the obvious case.
      msg =~
        ~r/(tool[_\s]?choice|tool[_\s]?call|function[_\s]?call|response[_\s]?format|json[_\s]?(schema|object|response))/i and
          msg =~ ~r/(not\s+(supported|allowed|valid)|unsupported|invalid|unrecognized|unknown)/i ->
        true

      # "Unrecognized request argument supplied: <key>" — happens when a
      # provider doesn't accept a JSON-schema field we sent (e.g. some
      # OpenAI-compat servers reject `description` on the parameters
      # object). Treat as a strategy mismatch so we fall back rather than
      # crash the call.
      msg =~ ~r/unrecognized\s+(request\s+)?argument/i ->
        true

      true ->
        false
    end
  end

  defp transient?(msg) do
    msg =~
      ~r/(timeout|timed\s*out|connection|unreachable|temporarily|rate[_\s]?limit|overloaded|503|504|429)/i
  end
end
