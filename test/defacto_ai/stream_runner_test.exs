defmodule DefactoAI.StreamRunnerTest do
  use ExUnit.Case, async: true

  alias DefactoAI.StreamRunner
  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message

  @error_body ~s({"error":{"code":422,"message":"failed to fetch image; check the url provided is valid","type":"unprocessable_entity"}})

  defp chain do
    llm =
      ChatOpenAI.new!(%{
        endpoint: "https://api.example.com/v1/chat/completions",
        api_key: "sk-test",
        model: "gpt-test",
        stream: true
      })

    LLMChain.new!(%{llm: llm})
    |> LLMChain.add_message(Message.new_user!("describe this image"))
  end

  defp sse_body(json_chunks) do
    json_chunks
    |> Enum.map(&"data: #{&1}\n\n")
    |> Enum.join()
    |> Kernel.<>("data: [DONE]\n\n")
  end

  describe "run/2" do
    test "assembles streamed content deltas into an assistant message" do
      Req.Test.stub(__MODULE__, fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_resp(
          200,
          sse_body([
            ~s({"choices":[{"delta":{"content":"Hello, "}}]}),
            ~s({"choices":[{"delta":{"content":"world!"}}]})
          ])
        )
      end)

      assert {:ok, %LLMChain{last_message: %Message{role: :assistant, content: "Hello, world!"}}} =
               StreamRunner.run(chain(), plug: {Req.Test, __MODULE__})
    end

    test "preserves the provider's error body on non-200 responses" do
      # Heroku Inference answers a bad image URL with a 422 and a JSON error
      # body. The SSE collector only reads `data:` lines, so without buffering
      # the raw body the caller would see {:api_error, 422, ""} and lose the
      # message it classifies on.
      Req.Test.stub(__MODULE__, fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.send_resp(422, @error_body)
      end)

      assert {:error, %LLMChain{}, {:api_error, 422, body}} =
               StreamRunner.run(chain(), plug: {Req.Test, __MODULE__})

      assert body == @error_body
      assert body =~ "failed to fetch image"
    end

    test "falls back to an empty body when the error response has no body" do
      Req.Test.stub(__MODULE__, fn conn -> Plug.Conn.send_resp(conn, 503, "") end)

      assert {:error, %LLMChain{}, {:api_error, 503, ""}} =
               StreamRunner.run(chain(), plug: {Req.Test, __MODULE__})
    end
  end
end
