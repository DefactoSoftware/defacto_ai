defmodule DefactoAI.LangChainAdapterTest do
  use ExUnit.Case, async: true

  alias DefactoAI.LangChainAdapter
  alias DefactoAI.TestProvider
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message

  describe "build_chat_model/2" do
    test "concatenates base_url and api_path into the endpoint" do
      chat = LangChainAdapter.build_chat_model(TestProvider.new())
      assert %ChatOpenAI{endpoint: "https://api.example.com/v1/chat/completions"} = chat
    end

    test "trims a trailing slash on base_url before joining" do
      chat = LangChainAdapter.build_chat_model(TestProvider.new(%{base_url: "https://api.example.com/"}))
      assert chat.endpoint == "https://api.example.com/v1/chat/completions"
    end

    test "passes through api_key, model, and a long receive_timeout" do
      chat = LangChainAdapter.build_chat_model(TestProvider.new())
      assert chat.api_key == "sk-test"
      assert chat.model == "gpt-test"
      assert chat.receive_timeout == 180_000
    end

    test "merges strategy-specific overrides into the base config" do
      chat = LangChainAdapter.build_chat_model(TestProvider.new(), json_response: true)
      assert chat.json_response == true
    end

    test "tool_choice override is preserved" do
      chat =
        LangChainAdapter.build_chat_model(TestProvider.new(),
          tool_choice: %{"type" => "function", "function" => %{"name" => "respond"}}
        )

      assert chat.tool_choice == %{"type" => "function", "function" => %{"name" => "respond"}}
    end

    test "defaults to non-streaming" do
      chat = LangChainAdapter.build_chat_model(TestProvider.new())
      refute chat.stream
    end

    test "explicit stream override wins over the Application env default" do
      chat = LangChainAdapter.build_chat_model(TestProvider.new(), stream: true)
      assert chat.stream
    end
  end

  describe "build_chain/3" do
    test "wraps a chat model and converts simple maps into Message structs" do
      chat = LangChainAdapter.build_chat_model(TestProvider.new())

      chain =
        LangChainAdapter.build_chain(chat, [
          %{role: "system", content: "be brief"},
          %{role: "user", content: "hello"}
        ])

      assert [%Message{role: :system}, %Message{role: :user, content: content}] = chain.messages
      assert IO.iodata_to_binary(content_to_text(content)) == "hello"
    end

    test "accepts pre-built Message structs" do
      chat = LangChainAdapter.build_chat_model(TestProvider.new())
      msg = Message.new_user!("ping")

      chain = LangChainAdapter.build_chain(chat, [msg])
      assert [^msg] = chain.messages
    end

    test "accepts string-keyed messages" do
      chat = LangChainAdapter.build_chat_model(TestProvider.new())

      chain =
        LangChainAdapter.build_chain(chat, [%{"role" => "user", "content" => "hi"}])

      assert [%Message{role: :user}] = chain.messages
    end
  end

  defp content_to_text(content) when is_binary(content), do: content

  defp content_to_text(parts) when is_list(parts) do
    Enum.map(parts, fn
      %LangChain.Message.ContentPart{type: :text, content: c} -> c
      _ -> ""
    end)
  end
end
