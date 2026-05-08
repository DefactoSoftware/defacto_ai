defmodule DefactoAI.Embeddings.ClientTest do
  use ExUnit.Case, async: false

  alias DefactoAI.Embeddings.Client
  alias DefactoAI.TestProvider

  @sensitive_text "Heb je vragen over deze cursus? Mail dan naar foo@example.com."

  setup do
    provider =
      TestProvider.new(%{
        base_url: "https://api.example.com",
        api_path: "/v1/embeddings",
        api_key: "secret",
        model: "text-embedding-3-small"
      })

    previous_resolver = Application.get_env(:defacto_ai, :provider_resolver)
    previous_reporter = Application.get_env(:defacto_ai, :error_reporter)

    Application.put_env(:defacto_ai, :provider_resolver, fn :embedding -> provider end)

    on_exit(fn ->
      if previous_resolver,
        do: Application.put_env(:defacto_ai, :provider_resolver, previous_resolver),
        else: Application.delete_env(:defacto_ai, :provider_resolver)

      if previous_reporter,
        do: Application.put_env(:defacto_ai, :error_reporter, previous_reporter),
        else: Application.delete_env(:defacto_ai, :error_reporter)
    end)

    Req.Test.set_req_test_to_private()
    {:ok, provider: provider}
  end

  defp capture_reports do
    test_pid = self()

    Application.put_env(:defacto_ai, :error_reporter, fn title, extras ->
      send(test_pid, {:error_reported, title, extras})
    end)

    :ok
  end

  describe "embed/2 — rate limiting" do
    test "429 returns a structured rate-limited error using the Retry-After header" do
      Req.Test.stub(__MODULE__, fn conn ->
        conn
        |> Plug.Conn.put_resp_header("retry-after", "12")
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.send_resp(429, ~s({"error":"rate limit"}))
      end)

      capture_reports()

      assert {:error, {:rate_limited, seconds}} =
               Client.embed(@sensitive_text, plug: {Req.Test, __MODULE__})

      # base 12s + jitter in [0, 30] inclusive of base.
      assert seconds >= 12
      assert seconds <= 12 + 30

      # 429 should not flood the error reporter — it's an expected coordination signal.
      refute_received {:error_reported, _, _}
    end

    test "429 without Retry-After falls back to a default snooze with jitter" do
      Req.Test.stub(__MODULE__, fn conn ->
        Plug.Conn.send_resp(conn, 429, "")
      end)

      assert {:error, {:rate_limited, seconds}} =
               Client.embed(@sensitive_text, plug: {Req.Test, __MODULE__})

      assert seconds >= 60
      assert seconds <= 90
    end

    test "non-numeric Retry-After (e.g. HTTP date) falls back to default" do
      Req.Test.stub(__MODULE__, fn conn ->
        conn
        |> Plug.Conn.put_resp_header("retry-after", "Wed, 21 Oct 2026 07:28:00 GMT")
        |> Plug.Conn.send_resp(429, "")
      end)

      assert {:error, {:rate_limited, seconds}} =
               Client.embed(@sensitive_text, plug: {Req.Test, __MODULE__})

      assert seconds >= 60
      assert seconds <= 90
    end
  end

  describe "embed/2 — error reporting" do
    test "non-429 API errors invoke the error_reporter with a stable title and content extras" do
      Req.Test.stub(__MODULE__, fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.send_resp(500, ~s({"error":"server error"}))
      end)

      capture_reports()

      assert {:error, {:api_error, 500, _body}} =
               Client.embed(@sensitive_text, plug: {Req.Test, __MODULE__})

      assert_received {:error_reported, "Embedding API error", extras}

      assert extras.status == 500
      assert extras.response_body == ~s({"error":"server error"})
      assert extras.content_length == String.length(@sensitive_text)
      assert extras.content == @sensitive_text
      assert extras.url == "https://api.example.com/v1/embeddings"
    end

    test "API error report title stays stable across different content lengths" do
      texts = ["short text", String.duplicate("longer ", 50)]

      Req.Test.stub(__MODULE__, fn conn ->
        Plug.Conn.send_resp(conn, 500, "boom")
      end)

      capture_reports()

      Enum.each(texts, fn text ->
        Client.embed(text, plug: {Req.Test, __MODULE__})
      end)

      assert_received {:error_reported, "Embedding API error", _}
      assert_received {:error_reported, "Embedding API error", _}
    end

    test "HTTP errors invoke the reporter with a stable title and reason in extras" do
      Req.Test.stub(__MODULE__, fn conn ->
        Req.Test.transport_error(conn, :timeout)
      end)

      capture_reports()

      assert {:error, {:http_error, :timeout}} =
               Client.embed(@sensitive_text, plug: {Req.Test, __MODULE__})

      assert_received {:error_reported, "Embedding HTTP error", extras}

      assert extras.reason == ":timeout"
      assert extras.url == "https://api.example.com/v1/embeddings"
      assert extras.content_length == String.length(@sensitive_text)
      assert extras.content == @sensitive_text
    end

    test "no error_reporter configured = no crash" do
      Req.Test.stub(__MODULE__, fn conn ->
        Plug.Conn.send_resp(conn, 500, "boom")
      end)

      Application.delete_env(:defacto_ai, :error_reporter)

      assert {:error, {:api_error, 500, _}} =
               Client.embed(@sensitive_text, plug: {Req.Test, __MODULE__})
    end
  end

  describe "embed/2 — happy path" do
    test "returns the embedding vector from a 200 response" do
      Req.Test.stub(__MODULE__, fn conn ->
        body =
          Jason.encode!(%{
            "data" => [%{"index" => 0, "embedding" => [0.1, 0.2, 0.3]}]
          })

        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.send_resp(200, body)
      end)

      assert {:ok, [0.1, 0.2, 0.3]} =
               Client.embed("hello world", plug: {Req.Test, __MODULE__})
    end

    test "uses an explicit :provider opt over the configured resolver", %{provider: provider} do
      Application.put_env(:defacto_ai, :provider_resolver, fn _ -> nil end)

      Req.Test.stub(__MODULE__, fn conn ->
        body = Jason.encode!(%{"data" => [%{"index" => 0, "embedding" => [1.0]}]})
        Plug.Conn.send_resp(conn, 200, body)
      end)

      assert {:ok, [1.0]} =
               Client.embed("hello", provider: provider, plug: {Req.Test, __MODULE__})
    end
  end
end
