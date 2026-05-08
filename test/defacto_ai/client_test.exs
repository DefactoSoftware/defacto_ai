defmodule DefactoAI.ClientTest do
  use ExUnit.Case, async: false

  alias DefactoAI.Client
  alias DefactoAI.Client.Stub

  defmodule TestSchema do
    use Ecto.Schema
    import Ecto.Changeset

    @primary_key false
    embedded_schema do
      field(:value, :string)
    end

    def changeset(struct, attrs, _opts \\ []), do: cast(struct, attrs, [:value])
  end

  setup do
    Stub.reset()

    previous = Application.get_env(:defacto_ai, :client)
    Application.put_env(:defacto_ai, :client, Stub)

    on_exit(fn ->
      if previous do
        Application.put_env(:defacto_ai, :client, previous)
      else
        Application.delete_env(:defacto_ai, :client)
      end
    end)

    :ok
  end

  test "dispatches to the configured implementation" do
    Stub.expect(TestSchema, fn _msgs, _opts -> {:ok, %TestSchema{value: "stubbed"}} end)

    assert {:ok, %TestSchema{value: "stubbed"}} =
             Client.complete_structured(TestSchema, [%{role: "user", content: "hi"}])
  end

  test "the stub raises when no expectation is registered" do
    assert_raise RuntimeError, ~r/no expectation set for/, fn ->
      Client.complete_structured(TestSchema, [%{role: "user", content: "hi"}])
    end
  end

  test "the stub passes through messages and opts to the response function" do
    Stub.expect(TestSchema, fn messages, opts ->
      {:ok, %TestSchema{value: "msgs:#{length(messages)} role:#{inspect(opts[:role])}"}}
    end)

    {:ok, %TestSchema{value: result}} =
      Client.complete_structured(TestSchema, [%{role: "user", content: "x"}], role: :summary)

    assert result == "msgs:1 role::summary"
  end

  test "default role is :llm when neither :provider nor :role is given" do
    Stub.expect(TestSchema, fn _msgs, opts -> {:ok, %TestSchema{value: inspect(opts[:role])}} end)

    {:ok, %TestSchema{value: role}} =
      Client.complete_structured(TestSchema, [%{role: "user", content: "x"}])

    assert role == ":llm"
  end

  test "default role is not added when an explicit :provider is given" do
    Stub.expect(TestSchema, fn _msgs, opts ->
      {:ok, %TestSchema{value: "role=#{inspect(opts[:role])} provider=#{inspect(opts[:provider])}"}}
    end)

    {:ok, %TestSchema{value: result}} =
      Client.complete_structured(TestSchema, [%{role: "user", content: "x"}], provider: :explicit)

    assert result == "role=nil provider=:explicit"
  end

  test "the stub finds expectations through $callers" do
    Stub.expect(TestSchema, fn _, _ -> {:ok, %TestSchema{value: "from-task"}} end)

    task =
      Task.async(fn ->
        Client.complete_structured(TestSchema, [%{role: "user", content: "x"}])
      end)

    assert {:ok, %TestSchema{value: "from-task"}} = Task.await(task)
  end

  test "the stub accepts a static tuple in addition to a function" do
    Stub.expect(TestSchema, {:ok, %TestSchema{value: "static"}})

    assert {:ok, %TestSchema{value: "static"}} =
             Client.complete_structured(TestSchema, [%{role: "user", content: "x"}])
  end

  describe "complete_chat/2" do
    test "returns the stub's full reply as a binary" do
      Stub.expect_chat(fn _msgs, _opts -> {:ok, "the answer"} end)

      assert {:ok, "the answer"} =
               Client.complete_chat([%{role: "user", content: "what?"}])
    end

    test "joins a list of streamed chunks into a single binary" do
      Stub.expect_chat(fn _msgs, _opts -> {:ok, ["Hello, ", "world!"]} end)

      assert {:ok, "Hello, world!"} =
               Client.complete_chat([%{role: "user", content: "x"}])
    end

    test "surfaces stub errors" do
      Stub.expect_chat({:error, :no_llm_provider})

      assert {:error, :no_llm_provider} =
               Client.complete_chat([%{role: "user", content: "x"}])
    end

    test "raises when no chat expectation is registered" do
      assert_raise RuntimeError, ~r/no chat expectation set/, fn ->
        Client.complete_chat([%{role: "user", content: "x"}])
      end
    end
  end

  describe "stream_chat/2" do
    test "returns a stream of chunks for a list response" do
      Stub.expect_chat(fn _msgs, _opts -> {:ok, ["a", "b", "c"]} end)

      assert {:ok, stream} = Client.stream_chat([%{role: "user", content: "x"}])
      assert Enum.to_list(stream) == ["a", "b", "c"]
    end

    test "wraps a single binary in a one-element stream" do
      Stub.expect_chat({:ok, "single chunk"})

      assert {:ok, stream} = Client.stream_chat([%{role: "user", content: "x"}])
      assert Enum.to_list(stream) == ["single chunk"]
    end

    test "emits errors as a single error element" do
      Stub.expect_chat({:error, :timeout})

      assert {:ok, stream} = Client.stream_chat([%{role: "user", content: "x"}])
      assert Enum.to_list(stream) == [{:error, :timeout}]
    end
  end

  describe "telemetry events" do
    test "emits :start and :stop spans for complete_structured" do
      Stub.expect(TestSchema, {:ok, %TestSchema{value: "ok"}})

      ref = make_ref()
      :telemetry.attach_many(
        "test-#{inspect(ref)}",
        [
          [:defacto_ai, :complete_structured, :start],
          [:defacto_ai, :complete_structured, :stop]
        ],
        fn event, measurements, metadata, _ ->
          send(self(), {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      Client.complete_structured(TestSchema, [%{role: "user", content: "x"}])

      assert_received {:telemetry, [:defacto_ai, :complete_structured, :start], _, %{schema: TestSchema}}

      assert_received {:telemetry, [:defacto_ai, :complete_structured, :stop], %{duration: _},
                       %{schema: TestSchema}}

      :telemetry.detach("test-#{inspect(ref)}")
    end
  end
end
