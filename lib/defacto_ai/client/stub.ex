defmodule DefactoAI.Client.Stub do
  @moduledoc """
  Test-only `DefactoAI.Client` implementation.

  Each test process registers expected responses keyed by schema module:

      DefactoAI.Client.Stub.expect(MyApp.SomeSchema, fn _messages, _opts ->
        {:ok, %MyApp.SomeSchema{field: "value"}}
      end)

  When `complete_structured/3` is called, the stub looks up the
  expectation for the schema in the current process's dictionary,
  walking `$callers` so that work spawned in Tasks (or run inline by
  Oban testing helpers) finds the expectation registered by the test.

  If no expectation is registered the stub raises — silent default
  responses would mask test gaps.

  Wire it up in `config/test.exs`:

      config :defacto_ai, :client, DefactoAI.Client.Stub

  Ships in `lib/` rather than `test/support/` so any consumer of
  `defacto_ai` can use it without copying the file or fiddling with
  `elixirc_paths`.
  """

  @behaviour DefactoAI.Client

  @key :defacto_ai_client_stub
  @chat_key :defacto_ai_client_stub_chat

  @type response_fun ::
          (messages :: [map()], opts :: keyword() -> {:ok, struct()} | {:error, term()})

  @doc """
  Register a response for `schema_module` in the current test process.

  `response` is either a `{:ok, struct}` / `{:error, reason}` tuple
  (returned as-is for every call) or a 2-arity function that receives
  the `messages` and `opts` and returns either of those tuples.
  """
  @spec expect(module(), {:ok, struct()} | {:error, term()} | response_fun()) :: :ok
  def expect(schema_module, response)
      when is_atom(schema_module) and (is_function(response, 2) or is_tuple(response)) do
    expectations = Process.get(@key, %{})
    Process.put(@key, Map.put(expectations, schema_module, response))
    :ok
  end

  @doc """
  Register a response for plain chat completion / streaming chat in the
  current test process.

  `response` is one of:
    * `{:ok, binary}` — a single full reply (used as-is for `complete_chat`,
      wrapped in a one-element stream for `stream_chat`).
    * `{:ok, [binary]}` — a list of chunks (joined for `complete_chat`,
      streamed in order for `stream_chat`).
    * `{:error, reason}` — surfaced as-is for `complete_chat`, emitted as
      a single error element for `stream_chat`.
    * a 2-arity function `(messages, opts -> any of the above)`.
  """
  @spec expect_chat(
          {:ok, binary() | [binary()]} | {:error, term()} | response_fun()
        ) :: :ok
  def expect_chat(response)
      when is_function(response, 2) or is_tuple(response) do
    Process.put(@chat_key, response)
    :ok
  end

  @doc """
  Clear all registered expectations for the current test process.
  """
  @spec reset() :: :ok
  def reset do
    Process.delete(@key)
    Process.delete(@chat_key)
    :ok
  end

  @impl true
  def complete_structured(schema_module, messages, opts \\ []) do
    case lookup(schema_module) do
      {:ok, fun} when is_function(fun, 2) ->
        fun.(messages, opts)

      {:ok, {:ok, _} = response} ->
        response

      {:ok, {:error, _} = response} ->
        response

      :error ->
        raise """
        DefactoAI.Client.Stub: no expectation set for #{inspect(schema_module)}.

        Register one in your test setup with:

            DefactoAI.Client.Stub.expect(#{inspect(schema_module)}, fn _msgs, _opts ->
              {:ok, %#{inspect(schema_module)}{...}}
            end)
        """
    end
  end

  @impl true
  def complete_chat(messages, opts \\ []) do
    case resolve_chat_response(messages, opts) do
      {:ok, str} when is_binary(str) -> {:ok, str}
      {:ok, chunks} when is_list(chunks) -> {:ok, Enum.join(chunks)}
      {:error, _} = error -> error
    end
  end

  @impl true
  def stream_chat(messages, opts \\ []) do
    case resolve_chat_response(messages, opts) do
      {:ok, str} when is_binary(str) -> {:ok, [str]}
      {:ok, chunks} when is_list(chunks) -> {:ok, chunks}
      {:error, _} = error -> {:ok, [error]}
    end
  end

  defp resolve_chat_response(messages, opts) do
    case lookup_chat() do
      {:ok, fun} when is_function(fun, 2) -> fun.(messages, opts)
      {:ok, response} when is_tuple(response) -> response
      :error -> raise_no_chat_expectation()
    end
  end

  defp raise_no_chat_expectation do
    raise """
    DefactoAI.Client.Stub: no chat expectation set.

    Register one in your test setup with:

        DefactoAI.Client.Stub.expect_chat(fn _msgs, _opts ->
          {:ok, "the answer"}             # full reply
          # or
          {:ok, ["streamed ", "chunks"]}  # streamed
          # or
          {:error, :timeout}
        end)
    """
  end

  defp lookup_chat do
    walk_callers(fn dict -> Map.fetch(dict, @chat_key) end)
  end

  defp lookup(schema_module) do
    walk_callers(fn dict ->
      case Map.get(dict, @key) do
        %{^schema_module => response} -> {:ok, response}
        _ -> :error
      end
    end)
  end

  defp walk_callers(fun) do
    Enum.reduce_while([self() | callers()], :error, fn pid, _acc ->
      dict = process_dict(pid)

      case fun.(dict) do
        {:ok, _} = found -> {:halt, found}
        :error -> {:cont, :error}
      end
    end)
  end

  defp callers do
    Process.get(:"$callers", [])
  end

  defp process_dict(pid) when pid == self() do
    Process.get()
    |> Enum.into(%{})
  end

  defp process_dict(pid) do
    case Process.info(pid, :dictionary) do
      {:dictionary, dict} -> Enum.into(dict, %{})
      nil -> %{}
    end
  end
end
