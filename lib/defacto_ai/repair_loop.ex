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

  @type decode_fun :: (LLMChain.t() -> {:ok, map() | binary()} | {:error, term()})
  @type result :: {:ok, struct()} | {:error, term()}

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

  # A forced tool call comes back as an assistant message with `nil` content
  # (the payload lives in `tool_calls`). When the repair loop re-sends that
  # turn on a corrective retry, LangChain serialises it as `"content": null`.
  # The OpenAI spec permits that alongside `tool_calls`, but some
  # OpenAI-compatible providers are stricter and reject it with a 400 like
  # `messages[1]: content is required`. Coerce `nil` content to "" so the
  # message history round-trips on those providers without losing the tool
  # calls or our retry budget.
  defp ensure_message_content(%LLMChain{messages: messages} = chain) do
    %{chain | messages: Enum.map(messages, &fill_blank_content/1)}
  end

  defp fill_blank_content(%Message{content: nil} = message), do: %{message | content: ""}
  defp fill_blank_content(%Message{} = message), do: message

  defp decode_and_validate(chain, schema_module, decode_fun, budget, opts) do
    case decode_fun.(chain) do
      {:ok, payload} ->
        case JSON.decode_and_cast(payload, schema_module, opts) do
          {:ok, struct} ->
            {:ok, struct}

          {:error, validation_error} when budget > 0 ->
            Logger.debug(fn ->
              "DefactoAI: validation failed, retrying with corrective message " <>
                "(#{budget - 1} attempts left)"
            end)

            :telemetry.execute(
              [:defacto_ai, :repair_loop, :retry],
              %{remaining: budget - 1},
              %{schema: schema_module}
            )

            chain
            |> LLMChain.add_message(corrective_message(validation_error, payload))
            |> run(schema_module, decode_fun, budget - 1, opts)

          {:error, validation_error} ->
            {:error, {:validation_failed, validation_error}}
        end

      {:error, decode_error} ->
        {:error, {:decode_failed, decode_error}}
    end
  end

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
