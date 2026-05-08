defmodule DefactoAI.Strategy do
  @moduledoc """
  Behaviour for the three structured-output strategies the client tries
  in order: tool calling, `response_format: json_object`, plain-text
  with repair.

  Each implementation knows how to:

  1. Build an `LLMChain` configured for that strategy (`prepare/4`).
  2. Pull the structured payload out of the chain's last message
     (`decode_payload/1`). The payload is either a map (when the
     provider already gave us parsed JSON, e.g. tool-call arguments)
     or a binary (raw text) that the orchestrator runs through
     `DefactoAI.JSON.decode_and_cast/3`.

  Strategy fallback (when the provider rejects this strategy entirely)
  lives in `DefactoAI.Client.LangChain` — the strategy itself just
  surfaces errors.
  """

  alias LangChain.Chains.LLMChain

  @type provider :: any()
  @type schema_module :: module()
  @type messages :: [map() | LangChain.Message.t()]
  @type opts :: keyword()
  @type payload :: map() | binary()
  @type prepare_error ::
          :unsupported
          | {:invalid_schema, term()}
          | term()
  @type decode_error ::
          :no_message
          | :no_content
          | :no_tool_call
          | :wrong_tool_call
          | term()

  @callback prepare(provider(), schema_module(), messages(), opts()) ::
              {:ok, LLMChain.t()} | {:error, prepare_error()}

  @callback decode_payload(LLMChain.t()) :: {:ok, payload()} | {:error, decode_error()}

  @doc """
  Default strategy order: prefer the most reliable form first.
  """
  @spec default_order() :: [module()]
  def default_order do
    [
      DefactoAI.Strategy.ToolCall,
      DefactoAI.Strategy.JsonMode,
      DefactoAI.Strategy.TextRepair
    ]
  end
end
