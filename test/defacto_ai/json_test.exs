defmodule DefactoAI.JSONTest do
  use ExUnit.Case, async: true

  defmodule TestSchema do
    use Ecto.Schema
    import Ecto.Changeset

    @primary_key false
    embedded_schema do
      field(:name, :string)
      field(:count, :integer)
    end

    def changeset(struct, attrs, opts \\ []) do
      required = if Keyword.get(opts, :name_required?, true), do: [:name], else: []

      struct
      |> cast(attrs, [:name, :count])
      |> validate_required(required)
    end
  end

  describe "extract/1" do
    test "decodes a clean JSON object" do
      assert {:ok, %{"a" => 1}} = DefactoAI.JSON.extract(~s({"a": 1}))
    end

    test "decodes a clean JSON array" do
      assert {:ok, [1, 2, 3]} = DefactoAI.JSON.extract(~s([1, 2, 3]))
    end

    test "strips ```json fences" do
      input = """
      ```json
      {"a": 1}
      ```
      """

      assert {:ok, %{"a" => 1}} = DefactoAI.JSON.extract(input)
    end

    test "strips bare ``` fences without a language tag" do
      input = """
      ```
      {"a": 1}
      ```
      """

      assert {:ok, %{"a" => 1}} = DefactoAI.JSON.extract(input)
    end

    test "slices JSON out of surrounding prose" do
      input = ~s|Sure! Here is the answer: {"answer": 42}. Hope this helps!|
      assert {:ok, %{"answer" => 42}} = DefactoAI.JSON.extract(input)
    end

    test "respects strings when balancing braces" do
      input = ~s|prefix {"text": "a } b { c"} suffix|
      assert {:ok, %{"text" => "a } b { c"}} = DefactoAI.JSON.extract(input)
    end

    test "respects escaped quotes inside strings" do
      input = ~s|{"text": "she said \\"hi\\""}|
      assert {:ok, %{"text" => "she said \"hi\""}} = DefactoAI.JSON.extract(input)
    end

    test "handles nested objects and arrays" do
      input = ~s|garbage {"a": {"b": [1, {"c": 2}]}} trailing|
      assert {:ok, %{"a" => %{"b" => [1, %{"c" => 2}]}}} = DefactoAI.JSON.extract(input)
    end

    test "returns :no_json when no JSON is present" do
      assert {:error, :no_json} = DefactoAI.JSON.extract("just words, no JSON here")
    end

    test "returns invalid_json when the slice fails to decode" do
      assert {:error, {:invalid_json, _}} = DefactoAI.JSON.extract(~s({"a": invalid}))
    end
  end

  describe "repair/1" do
    test "removes trailing commas before closing braces" do
      assert {:ok, %{"a" => 1, "b" => 2}} = DefactoAI.JSON.repair(~s({"a": 1, "b": 2,}))
    end

    test "removes trailing commas before closing brackets" do
      assert {:ok, [1, 2, 3]} = DefactoAI.JSON.repair(~s([1, 2, 3,]))
    end

    test "normalises smart double quotes" do
      input = "{“a”: “b”}"
      assert {:ok, %{"a" => "b"}} = DefactoAI.JSON.repair(input)
    end

    test "still works on already-clean JSON" do
      assert {:ok, %{"a" => 1}} = DefactoAI.JSON.repair(~s({"a": 1}))
    end

    test "returns an error when the payload cannot be salvaged" do
      assert {:error, _} = DefactoAI.JSON.repair("not json at all")
    end
  end

  describe "cast/3" do
    test "casts a map into the schema" do
      assert {:ok, %TestSchema{name: "x", count: 3}} =
               DefactoAI.JSON.cast(%{"name" => "x", "count" => 3}, TestSchema)
    end

    test "returns a changeset on validation failure" do
      assert {:error, %Ecto.Changeset{valid?: false}} =
               DefactoAI.JSON.cast(%{}, TestSchema)
    end

    test "passes opts through to changeset/3" do
      assert {:ok, %TestSchema{name: nil}} =
               DefactoAI.JSON.cast(%{}, TestSchema, name_required?: false)
    end
  end

  describe "decode_and_cast/3" do
    test "decodes binary input and casts in one call" do
      assert {:ok, %TestSchema{name: "x", count: 3}} =
               DefactoAI.JSON.decode_and_cast(~s({"name": "x", "count": 3}), TestSchema)
    end

    test "skips extract/repair when input is already a map" do
      assert {:ok, %TestSchema{name: "x"}} =
               DefactoAI.JSON.decode_and_cast(%{"name" => "x"}, TestSchema)
    end

    test "falls back to repair when extract fails on fixable input" do
      assert {:ok, %TestSchema{name: "x", count: 1}} =
               DefactoAI.JSON.decode_and_cast(
                 ~s|prefix {"name": "x", "count": 1,} trailing|,
                 TestSchema
               )
    end

    test "returns the original extract error when repair also fails" do
      assert {:error, {:invalid_json, _}} =
               DefactoAI.JSON.decode_and_cast(~s({"a": invalid}), TestSchema)
    end

    test "returns :no_json when text contains no JSON at all" do
      assert {:error, :no_json} =
               DefactoAI.JSON.decode_and_cast("nothing to see here", TestSchema)
    end
  end
end
