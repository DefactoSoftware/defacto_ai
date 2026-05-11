defmodule DefactoAI.Embeddings.Embeddable do
  @moduledoc """
  Behaviour for schemas that can have embeddings generated.

  ## Usage

      defmodule MyApp.Article do
        use DefactoAI.Embeddings.Embeddable

        @impl DefactoAI.Embeddings.Embeddable
        def embedding_content(%{title: title, content: content, summary: summary}) do
          [title, summary, content]
          |> Enum.reject(&is_nil/1)
          |> Enum.join("\\n\\n")
        end

        @impl DefactoAI.Embeddings.Embeddable
        def embeddable_type, do: :article

        @impl DefactoAI.Embeddings.Embeddable
        def embedding_trigger_fields, do: [:title, :content, :summary]

        # Optional: override for chunked content
        @impl DefactoAI.Embeddings.Embeddable
        def chunked?, do: true
      end

  HTML / Markdown stripping and other content normalisation is the
  host's responsibility — implement it in your `embedding_content/1`
  callback. Keeping that out of the library means consumers don't have
  to pull in Floki or any other parser they don't already use.
  """

  @doc """
  Returns the text content to be embedded for this record.
  """
  @callback embedding_content(struct :: struct()) :: String.t()

  @doc """
  Returns the atom identifier for this embeddable type.
  """
  @callback embeddable_type() :: atom()

  @doc """
  Returns the list of fields that trigger re-embedding when changed.
  """
  @callback embedding_trigger_fields() :: [atom()]

  @doc """
  Returns whether this embeddable should be chunked before embedding.
  Defaults to false.
  """
  @callback chunked?() :: boolean()

  @optional_callbacks chunked?: 0

  defmacro __using__(_opts) do
    quote do
      @behaviour DefactoAI.Embeddings.Embeddable

      @doc false
      def chunked?, do: false

      defoverridable chunked?: 0
    end
  end
end
