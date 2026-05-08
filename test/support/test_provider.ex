defmodule DefactoAI.TestProvider do
  @moduledoc """
  Plain-struct provider used by the library's own tests. Stands in for
  whatever Ecto schema a host app implements `DefactoAI.Provider` on.
  """

  defstruct [
    :api_key,
    :base_url,
    :api_path,
    :model,
    :provider_type
  ]

  def new(attrs \\ %{}) do
    defaults = %{
      api_key: "sk-test",
      base_url: "https://api.example.com",
      api_path: "/v1/chat/completions",
      model: "gpt-test",
      provider_type: "openai"
    }

    struct(__MODULE__, Map.merge(defaults, Map.new(attrs)))
  end
end

defimpl DefactoAI.Provider, for: DefactoAI.TestProvider do
  def api_key(p), do: p.api_key
  def base_url(p), do: p.base_url
  def api_path(p), do: p.api_path
  def model(p), do: p.model
  def provider_type(p), do: p.provider_type
end
