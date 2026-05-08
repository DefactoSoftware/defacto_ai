defprotocol DefactoAI.Provider do
  @moduledoc """
  Configuration contract for an LLM / embeddings provider.

  Host applications implement this protocol on their own provider struct
  (typically an Ecto schema) so that the rest of the library can consume
  any host's provider data without knowing about the host's domain.

  ## Required fields

    * `api_key/1` — Bearer token used for `Authorization`.
    * `base_url/1` — origin of the provider, e.g. `"https://api.openai.com"`.
      A trailing slash is tolerated; the library trims it.
    * `api_path/1` — path component for chat completions, e.g.
      `"/v1/chat/completions"`. Returned as-is from the protocol; the
      library concatenates it with `base_url/1` to form the chat endpoint.
    * `model/1` — model identifier the provider expects, e.g. `"gpt-4o"`.
    * `provider_type/1` — free-form label (`"openai"`, `"anthropic-proxy"`,
      `"heroku-inference"`, …). Used for telemetry and log tagging only;
      may be `nil`.

  Embeddings always go through `<base_url>/v1/embeddings` regardless of
  `api_path/1` — `api_path/1` is for chat only.

  ## Example

      defmodule MyApp.AI.Provider do
        use Ecto.Schema

        schema "ai_providers" do
          field :api_key, :string
          field :base_url, :string
          field :api_path, :string, default: "/v1/chat/completions"
          field :model, :string
          field :provider_type, :string
        end
      end

      defimpl DefactoAI.Provider, for: MyApp.AI.Provider do
        def api_key(p), do: p.api_key
        def base_url(p), do: p.base_url
        def api_path(p), do: p.api_path
        def model(p), do: p.model
        def provider_type(p), do: p.provider_type
      end
  """

  @spec api_key(t) :: String.t()
  def api_key(provider)

  @spec base_url(t) :: String.t()
  def base_url(provider)

  @spec api_path(t) :: String.t()
  def api_path(provider)

  @spec model(t) :: String.t()
  def model(provider)

  @spec provider_type(t) :: String.t() | nil
  def provider_type(provider)
end
