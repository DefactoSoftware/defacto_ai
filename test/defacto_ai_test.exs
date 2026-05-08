defmodule DefactoAITest do
  use ExUnit.Case, async: true

  test "module compiles" do
    assert Code.ensure_loaded?(DefactoAI)
  end
end
