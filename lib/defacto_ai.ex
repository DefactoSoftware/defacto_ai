defmodule DefactoAI do
  @moduledoc """
  Shared building blocks for LLM, embeddings and similarity-search work
  across Defacto apps.

  See the README for installation and configuration.

  Public entry points:

    * `DefactoAI.Client` — structured outputs (`complete_structured/3`),
      plain chat (`complete_chat/2`), streaming chat (`stream_chat/2`).
    * `DefactoAI.Embeddings` — embedding generation, storage, deletion.
    * `DefactoAI.SimilaritySearch` — decay-weighted nearest-neighbour
      search over stored embeddings.
  """
end
