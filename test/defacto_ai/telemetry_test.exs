defmodule DefactoAI.TelemetryTest do
  use ExUnit.Case, async: true

  alias DefactoAI.Telemetry
  alias DefactoAI.TestProvider
  alias LangChain.TokenUsage

  test "usage is nil when nothing was reported" do
    {_, context} = Telemetry.track(fn -> :ok end)

    assert Telemetry.info(context).usage == %{input_tokens: nil, output_tokens: nil}
  end

  test "usage is summed over several rounds" do
    {_, context} =
      Telemetry.track(fn ->
        Telemetry.record_usage(10, 4)
        Telemetry.record_usage(%TokenUsage{input: 5, output: 1})
        Telemetry.record_usage(nil)
      end)

    assert Telemetry.info(context).usage == %{input_tokens: 15, output_tokens: 5}
  end

  test "a zero count is reported as zero, not nil" do
    {_, context} = Telemetry.track(fn -> Telemetry.record_usage(0, nil) end)

    assert Telemetry.info(context).usage == %{input_tokens: 0, output_tokens: nil}
  end

  test "records provider type and model as strings" do
    {_, context} =
      Telemetry.track(fn ->
        Telemetry.record_provider(TestProvider.new(%{provider_type: "acme", model: "m-1"}))
      end)

    assert %{provider: "acme", model: "m-1"} = Telemetry.info(context)
  end

  test "usage can be reported from another process holding the context" do
    {_, context} =
      Telemetry.track(fn ->
        ctx = Telemetry.current()

        Task.async(fn ->
          Telemetry.attach(ctx)
          Telemetry.record_usage(3, 2)
        end)
        |> Task.await()
      end)

    assert Telemetry.info(context).usage == %{input_tokens: 3, output_tokens: 2}
  end

  test "contexts nest and the outer one is restored" do
    {_, outer} =
      Telemetry.track(fn ->
        Telemetry.record_usage(1, 1)
        {_, inner} = Telemetry.track(fn -> Telemetry.record_usage(7, 7) end)
        Telemetry.record_usage(1, 1)
        assert Telemetry.info(inner).usage.input_tokens == 7
      end)

    assert Telemetry.info(outer).usage == %{input_tokens: 2, output_tokens: 2}
    assert Telemetry.current() == nil
  end

  test "recording outside a context is a no-op" do
    assert Telemetry.record_usage(1, 1) == :ok
    assert Telemetry.record_provider(TestProvider.new()) == :ok
  end
end
