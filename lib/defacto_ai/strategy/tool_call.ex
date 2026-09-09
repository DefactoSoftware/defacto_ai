defmodule DefactoAI.Strategy.ToolCall do
  @moduledoc """
  Structured-output strategy that uses tool / function calling.

  Registers a single tool named `"respond"` whose `parameters_schema` is
  the response module's `parameters_schema/0`. `tool_choice` forces the
  model to call that tool, so the assistant message comes back with a
  parsed-arguments map we can hand straight to the schema's changeset.

  The widest-reach strategy: tool calling is supported by virtually every
  major OpenAI-compatible provider (OpenAI, Anthropic via proxy, vLLM,
  Together, DeepSeek, Mistral, Gemini's OpenAI-compat endpoint, ...).
  """

  @behaviour DefactoAI.Strategy

  alias DefactoAI.LangChainAdapter
  alias LangChain.Function

  @tool_name "respond"

  @impl true
  def prepare(provider, schema_module, messages, opts) do
    with {:ok, parameters_schema} <- fetch_parameters_schema(schema_module) do
      tool =
        Function.new!(%{
          name: @tool_name,
          description: tool_description(schema_module, parameters_schema),
          parameters_schema: sanitise_for_function_parameters(parameters_schema),
          # The chain runs in single-step mode and never executes the tool —
          # we only read the parsed arguments back off the assistant message.
          # LangChain's Function requires a 2-arity callback to validate, so
          # we provide a no-op.
          function: fn _args, _ctx -> {:ok, "noop"} end
        })

      chat_model =
        LangChainAdapter.build_chat_model(
          provider,
          [tool_choice: %{"type" => "function", "function" => %{"name" => @tool_name}}],
          opts
        )

      chain =
        chat_model
        |> LangChainAdapter.build_chain(messages)
        |> LangChain.Chains.LLMChain.add_tools([tool])

      {:ok, chain}
    end
  end

  @impl true
  def decode_payload(%LangChain.Chains.LLMChain{last_message: nil}), do: {:error, :no_message}

  def decode_payload(%LangChain.Chains.LLMChain{last_message: message}) do
    case message.tool_calls || [] do
      [] -> {:error, :no_tool_call}
      calls -> calls |> pick_tool_call() |> extract_arguments()
    end
  end

  # Streaming gateways can leave a stray tool call next to the real one (an
  # opening chunk whose arguments landed elsewhere, or a second call the
  # model added). Prefer the `respond` call that actually carries arguments,
  # then any `respond` call, then whatever came first.
  defp pick_tool_call(calls) do
    Enum.find(calls, &(respond?(&1) and is_map(&1.arguments) and map_size(&1.arguments) > 0)) ||
      Enum.find(calls, &respond?/1) ||
      hd(calls)
  end

  defp respond?(%LangChain.Message.ToolCall{name: @tool_name}), do: true
  defp respond?(_), do: false

  defp extract_arguments(%LangChain.Message.ToolCall{name: @tool_name, arguments: args})
       when is_map(args),
       do: {:ok, args}

  defp extract_arguments(%LangChain.Message.ToolCall{name: @tool_name, arguments: nil}),
    do: {:ok, %{}}

  defp extract_arguments(%LangChain.Message.ToolCall{}), do: {:error, :wrong_tool_call}

  defp fetch_parameters_schema(schema_module) do
    Code.ensure_loaded(schema_module)

    if function_exported?(schema_module, :parameters_schema, 0) do
      {:ok, schema_module.parameters_schema()}
    else
      {:error, {:invalid_schema, {schema_module, :parameters_schema, 0}}}
    end
  end

  defp tool_description(schema_module, parameters_schema) do
    Code.ensure_loaded(schema_module)

    cond do
      function_exported?(schema_module, :tool_description, 0) ->
        schema_module.tool_description()

      is_map(parameters_schema) and is_binary(parameters_schema[:description]) ->
        parameters_schema[:description]

      is_map(parameters_schema) and is_binary(parameters_schema["description"]) ->
        parameters_schema["description"]

      true ->
        "Return the structured response as the arguments to this function."
    end
  end

  # Some OpenAI-compatible providers reject `description` (and a few other
  # keys) on the top-level parameters object, even though JSON Schema allows
  # them. Strip those before sending; descriptions on individual properties
  # are preserved.
  defp sanitise_for_function_parameters(schema) when is_map(schema) do
    schema
    |> Map.drop([:description, "description"])
  end

  defp sanitise_for_function_parameters(other), do: other
end
