defmodule DefactoAI.Embeddings.Client do
  @moduledoc """
  HTTP client for generating embeddings via OpenAI-compatible APIs.

  Uses Req under the hood. Public API:

    * `embed/2` returns `{:ok, embedding_vector}` or `{:error, reason}`.
    * `embed_batch/2` processes each text individually (so we don't trip
      provider per-batch limits) and returns `{:ok, [embedding]}`.

  Provider resolution mirrors `DefactoAI.Client`: if `opts[:provider]`
  is set it wins; otherwise the role-based resolver is consulted with
  `:embedding`. Configure the resolver:

      config :defacto_ai, provider_resolver: &MyApp.AI.resolve_provider/1

  ## Errors

    * `429`s come back as `{:error, {:rate_limited, seconds}}` honouring
      `Retry-After` with jitter so multiple instances don't synchronise
      retries.
    * Non-`200` non-`429` responses surface as
      `{:error, {:api_error, status, response_text}}`.
    * Transport errors surface as `{:error, {:http_error, reason}}`.

  ## Observability

    * `DefactoAI.ErrorReporter.report/2` is invoked for HTTP / API errors
      with stable titles (`"Embedding API error"` / `"Embedding HTTP error"`)
      so issues group by class in Sentry-style tools.
    * `Logger.warning/1` is used unconditionally so events show up even
      when no reporter is configured.

  Tests can stub the underlying HTTP layer with `Req.Test.stub/2` (passed
  via the `:plug` option).
  """

  require Logger

  alias DefactoAI.Client.LangChain, as: LangChainClient
  alias DefactoAI.Embeddings.RateLimiter
  alias DefactoAI.ErrorReporter
  alias DefactoAI.Provider

  @default_timeout 60_000

  @rate_limit_default_seconds 60
  @rate_limit_jitter_seconds 30

  @doc """
  Generate an embedding for a single text.
  """
  def embed(text, opts \\ []) do
    case embed_batch([text], opts) do
      {:ok, [embedding]} -> {:ok, embedding}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Generate embeddings for multiple texts, processing each individually.
  """
  def embed_batch(texts, opts \\ []) when is_list(texts) do
    opts = Keyword.put_new(opts, :role, :embedding)

    with {:ok, provider} <- LangChainClient.resolve_provider(opts) do
      results =
        Enum.reduce_while(texts, {:ok, []}, fn text, {:ok, acc} ->
          case do_embed_single(text, provider, opts) do
            {:ok, embedding} -> {:cont, {:ok, [embedding | acc]}}
            {:error, reason} -> {:halt, {:error, reason}}
          end
        end)

      case results do
        {:ok, embeddings} -> {:ok, Enum.reverse(embeddings)}
        error -> error
      end
    end
  end

  defp do_embed_single(text, provider, opts) do
    sanitized_text = sanitize_text(text)

    if sanitized_text == "" do
      Logger.warning("DefactoAI: skipping empty text for embedding")
      {:ok, List.duplicate(0.0, 1024)}
    else
      RateLimiter.acquire()

      Logger.debug(
        "DefactoAI: embedding text (#{String.length(sanitized_text)} chars): " <>
          "#{String.slice(sanitized_text, 0, 100)}..."
      )

      url = build_url(provider)
      timeout = Keyword.get(opts, :timeout, @default_timeout)

      req_opts =
        [
          url: url,
          json: %{model: Provider.model(provider), input: [sanitized_text]},
          auth: {:bearer, Provider.api_key(provider)},
          receive_timeout: timeout,
          retry: false,
          decode_body: true
        ]
        |> maybe_put(:plug, Keyword.get(opts, :plug) || default_plug())

      case Req.post(req_opts) do
        {:ok, %Req.Response{status: 200, body: body}} ->
          case parse_success(body) do
            {:ok, [embedding]} -> {:ok, embedding}
            {:ok, _} -> {:error, :unexpected_response}
            error -> error
          end

        {:ok, %Req.Response{status: 429, headers: headers}} ->
          seconds = rate_limit_wait_seconds(headers)
          Logger.warning("DefactoAI: embedding API rate limited — backing off for #{seconds}s")
          {:error, {:rate_limited, seconds}}

        {:ok, %Req.Response{status: status, body: body}} ->
          response_text = body_to_text(body)
          report_api_error(status, response_text, sanitized_text, url)
          {:error, {:api_error, status, response_text}}

        {:error, exception} ->
          reason = http_error_reason(exception)
          report_http_error(reason, sanitized_text, url)
          {:error, {:http_error, reason}}
      end
    end
  end

  defp report_api_error(status, response_body, sanitized_text, url) do
    Logger.warning("DefactoAI: embedding API error: status=#{status}")

    ErrorReporter.report("Embedding API error", %{
      status: status,
      url: url,
      response_body: response_body,
      content_length: String.length(sanitized_text),
      content: sanitized_text
    })
  end

  defp report_http_error(reason, sanitized_text, url) do
    Logger.warning("DefactoAI: embedding HTTP error: #{inspect(reason)}")

    ErrorReporter.report("Embedding HTTP error", %{
      reason: inspect(reason),
      url: url,
      content_length: String.length(sanitized_text),
      content: sanitized_text
    })
  end

  defp rate_limit_wait_seconds(headers) do
    base = retry_after_seconds(headers) || @rate_limit_default_seconds
    base + :rand.uniform(@rate_limit_jitter_seconds + 1) - 1
  end

  defp retry_after_seconds(headers) do
    Enum.find_value(headers, &parse_retry_after_header/1)
  end

  # Req normalises headers to `{name, [value]}` lists.
  defp parse_retry_after_header({name, values}) when is_list(values) do
    if String.downcase(name) == "retry-after" do
      Enum.find_value(values, &parse_non_negative_integer/1)
    end
  end

  defp parse_retry_after_header({name, value}) when is_binary(value) do
    if String.downcase(name) == "retry-after", do: parse_non_negative_integer(value)
  end

  defp parse_non_negative_integer(value) do
    case Integer.parse(value) do
      {seconds, ""} when seconds >= 0 -> seconds
      _ -> nil
    end
  end

  defp build_url(provider) do
    Provider.base_url(provider)
    |> String.trim_trailing("/")
    |> Kernel.<>("/v1/embeddings")
  end

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)

  defp default_plug do
    Application.get_env(:defacto_ai, :embeddings_req_plug)
  end

  defp parse_success(body) when is_map(body) do
    case body do
      %{"data" => data} ->
        embeddings =
          data
          |> Enum.sort_by(& &1["index"])
          |> Enum.map(& &1["embedding"])

        {:ok, embeddings}

      %{"error" => error} ->
        {:error, {:api_error, error}}

      _ ->
        {:error, :unexpected_response}
    end
  end

  defp parse_success(body) when is_binary(body) do
    case Jason.decode(body) do
      {:ok, decoded} -> parse_success(decoded)
      {:error, reason} -> {:error, {:json_decode_error, reason}}
    end
  end

  # When Req decodes JSON it returns a map; if not, it's a binary.
  defp body_to_text(body) when is_binary(body), do: body
  defp body_to_text(body), do: Jason.encode!(body)

  defp http_error_reason(%Req.TransportError{reason: reason}), do: reason
  defp http_error_reason(%Mint.TransportError{reason: reason}), do: reason
  defp http_error_reason(%{reason: reason}), do: reason
  defp http_error_reason(other), do: other

  defp sanitize_text(nil), do: ""

  defp sanitize_text(text) do
    text
    |> String.replace(~r/\[([^\]]+)\]\([^)]+\)/, "\\1")
    |> String.replace(~r/https?:\/\/[^\s]+/, "")
    |> String.replace(~r/\*+/, "")
    |> String.replace(~r/_+/, " ")
    |> String.replace(~r/^#+\s*/m, "")
    |> String.replace(~r/`+[^`]*`+/, "")
    |> String.replace(<<0>>, "")
    |> String.replace(~r/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/, "")
    |> String.normalize(:nfc)
    |> String.replace(~r/\p{Cs}/u, "")
    |> String.replace(~r/[^\p{L}\p{N}\p{P}\s]/u, "")
    |> String.replace(~r/[ \t]+/, " ")
    |> String.replace(~r/\n{3,}/, "\n\n")
    |> String.trim()
  end
end
