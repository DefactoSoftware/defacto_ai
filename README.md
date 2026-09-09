# DefactoAI

A small, opinionated Elixir library for taking structured output from
OpenAI-compatible LLMs, generating and storing embeddings, and running
similarity search over them.

Built around three ideas:

  - **Strategy fallback** for structured output — try tool calling first,
    fall back to JSON mode, then plain-text-with-repair, so the same
    response schema works against providers with very different feature
    sets.
  - **JSON repair + validation retry** — strip code fences, repair
    trailing commas / smart quotes, slice JSON out of prose, then cast
    through an Ecto changeset; on validation failure, append a corrective
    user message and re-run.
  - **Provider as a protocol, not a struct** — consumers implement
    `DefactoAI.Provider` on their own Ecto schema (or any struct), so the
    library never owns the host app's provider data.

## What's in the box

| Concern | Module |
|---|---|
| Structured / chat / streaming completions | `DefactoAI.Client` |
| Strategy fallback (tool call → JSON mode → text repair) | `DefactoAI.Strategy` |
| LLM-output JSON repair pipeline | `DefactoAI.JSON` |
| Validation-failure repair loop | `DefactoAI.RepairLoop` |
| LangChain `ChatOpenAI` adapter | `DefactoAI.LangChainAdapter` |
| Provider config contract | `DefactoAI.Provider` (protocol) |
| Embeddings HTTP client | `DefactoAI.Embeddings.Client` |
| Embeddings storage + orchestration | `DefactoAI.Embeddings`, `DefactoAI.Embedding` |
| Vector similarity search | `DefactoAI.SimilaritySearch` |
| Embeddable behaviour | `DefactoAI.Embeddings.Embeddable` |
| Test stub | `DefactoAI.Client.Stub` |
| Migration generator | `mix defacto_ai.gen.migration` |

## Installation

```elixir
# in mix.exs
def deps do
  [
    {:defacto_ai, github: "defacto-software/defacto_ai", ref: "<sha>"}
  ]
end
```

If you use the embeddings / similarity-search modules, also add
`pgvector`:

```elixir
{:pgvector, "~> 0.3"}
```

## Configuration

Minimum config:

```elixir
config :defacto_ai,
  repo: MyApp.Repo,
  provider_resolver: &MyApp.AI.resolve_provider/1
```

All keys:

| Key | Default | Purpose |
|---|---|---|
| `:repo` | — (required for embeddings) | Host app's `Ecto.Repo`. |
| `:provider_resolver` | — (required unless every call passes `provider:`) | 1-arity function `(role -> struct)`. Returns a struct that implements `DefactoAI.Provider`, or `nil` if no provider is configured for that role. |
| `:client` | `DefactoAI.Client.LangChain` | Implementation of the `Client` behaviour. Tests set this to `DefactoAI.Client.Stub`. |
| `:chat_stream` | `false` | Whether `LangChainAdapter` builds streaming chat models by default. Flip to `true` if your gateway times out long non-streamed completions. |
| `:default_strategies` | `[ToolCall, JsonMode, TextRepair]` | Order of structured-output strategies to try. |
| `:error_reporter` | `nil` | 2-arity function `(title, extras_map -> :ok)`. Called when the embeddings client encounters HTTP/API errors. Useful for Sentry integration; safe to leave unset. |
| `:embedding_rate_limit` | `450` | Embeddings per minute. |
| `:embeddings_req_plug` | `nil` | Test override — a `{Req.Test, MyTestModule}` tuple to stub the embeddings HTTP layer without passing it per-call. |
| `:prompts_app` | `nil` | OTP app that owns the `priv/prompts/` directory `DefactoAI.PromptRenderer` reads from. |
| `:prompts_dir` | `"prompts"` | Subdirectory under `priv/` for prompt templates. |

## Provider protocol

The library doesn't define a provider struct — host apps provide one (typically
an Ecto schema) and implement `DefactoAI.Provider` on it:

```elixir
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
```

## Provider resolution

Each call to `Client.complete_structured/3`, `Client.complete_chat/2`,
`Client.stream_chat/2`, `Embeddings.embed/2` etc. resolves a provider in
this order:

1. **Explicit struct** — `provider: %MyApp.AI.Provider{...}` in opts wins.
2. **Role lookup** — falls back to `opts[:role]` (defaults to `:llm` for
   chat-style calls, `:embedding` for the embeddings client) and calls
   the configured `provider_resolver`:

   ```elixir
   defmodule MyApp.AI do
     def resolve_provider(:llm), do: MyApp.AI.Providers.default_llm_provider()
     def resolve_provider(:embedding), do: MyApp.AI.Providers.default_embeddings_provider()
     def resolve_provider(:summary), do: MyApp.AI.Providers.default_summary_provider()
   end
   ```

The library ships three canonical roles (`:llm`, `:embedding`, `:summary`)
but the resolver is opaque — hosts can register additional role atoms
and pass them via `role:` if needed.

## Per-call options

`Client.complete_structured/3`, `Client.complete_chat/2` and
`Client.stream_chat/2` accept a keyword list of options:

| Option | Applies to | Description |
|---|---|---|
| `:provider` | all | Struct implementing `DefactoAI.Provider`. Wins over `:role`. |
| `:role` | all | Role handed to the configured `provider_resolver` (default `:llm`). |
| `:max_validation_retries` | `complete_structured` | Corrective retries per strategy before giving up (default `2`). |
| `:strategies` | `complete_structured` | Strategy order for this call, overriding `:default_strategies`. |
| `:validation_context` | `complete_structured` | Map/keyword lifted into the schema's `changeset/3` opts. |
| `:chat_model` | all | Keyword list or map of `LangChain.ChatModels.ChatOpenAI` attributes merged into every request, e.g. `chat_model: [max_tokens: 8_000]`. Use it when a gateway's default `max_tokens` truncates long structured answers (a cut-off tool-call JSON fails to parse and falls through every strategy). Strategy-controlled attributes (`tool_choice`, `json_response`, `stream` for `complete_chat`) always win. `stream_chat/2` only honours `:max_tokens` and `:temperature`. |

## Schema contract for structured outputs

Modules passed as the first argument to `Client.complete_structured/3`
must export:

* `changeset/2` or `changeset/3` (required) — Ecto changeset that casts
  and validates a parsed map. The 3-arity form receives the call's full
  opts so schemas can pull `:validation_context` data into validations.
* `parameters_schema/0` (recommended) — JSON-schema map registered as
  the `respond` tool's parameters and embedded into JSON-mode /
  text-repair system prompts.
* `tool_description/0` (optional) — string description used as the
  function tool's `description`. If omitted, falls back to the
  `description` field of `parameters_schema/0` if present.

Example:

```elixir
defmodule MyApp.AI.QuizResponse do
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    field :title, :string
    embeds_many :questions, MyApp.AI.QuizResponse.Question
  end

  def changeset(struct, attrs, _opts \\ []) do
    struct
    |> cast(attrs, [:title])
    |> cast_embed(:questions, required: true)
    |> validate_required([:title])
  end

  def parameters_schema do
    %{
      type: "object",
      properties: %{
        title: %{type: "string"},
        questions: %{type: "array", items: MyApp.AI.QuizResponse.Question.parameters_schema()}
      },
      required: ["title", "questions"]
    }
  end

  def tool_description, do: "Return a quiz matching the given content."
end
```

## Calling the LLM

```elixir
# Structured output (defaults to role: :llm)
{:ok, %MyApp.AI.QuizResponse{} = quiz} =
  DefactoAI.Client.complete_structured(MyApp.AI.QuizResponse, [
    %{role: "system", content: "You write quizzes."},
    %{role: "user", content: "5 questions about Erlang."}
  ])

# Plain chat
{:ok, "..."} = DefactoAI.Client.complete_chat([%{role: "user", content: "hi"}])

# Streaming chat
{:ok, stream} = DefactoAI.Client.stream_chat([%{role: "user", content: "hi"}])

stream
|> Stream.each(fn
  chunk when is_binary(chunk) -> IO.write(chunk)
  {:error, reason} -> IO.warn("stream failed: #{inspect(reason)}")
end)
|> Stream.run()

# Override the role (e.g. to use a cheaper summary provider)
DefactoAI.Client.complete_structured(MyApp.AI.SummaryResponse, msgs, role: :summary)

# Bypass the resolver entirely
DefactoAI.Client.complete_chat(msgs, provider: %MyApp.AI.Provider{...})
```

## Embeddings + similarity search

```bash
# One-time, in the host app:
mix defacto_ai.gen.migration
mix ecto.migrate
```

```elixir
# Implement Embeddable on your domain schemas:
defmodule MyApp.Article do
  use Ecto.Schema
  use DefactoAI.Embeddings.Embeddable

  schema "articles" do
    field :title, :string
    field :body, :string
  end

  @impl DefactoAI.Embeddings.Embeddable
  def embeddable_type, do: :article

  @impl DefactoAI.Embeddings.Embeddable
  def embedding_trigger_fields, do: [:title, :body]

  @impl DefactoAI.Embeddings.Embeddable
  def embedding_content(%{title: title, body: body}), do: "#{title}\n\n#{body}"

  @impl DefactoAI.Embeddings.Embeddable
  def chunked?, do: true
end

# Generate + store:
{:ok, _count} = DefactoAI.Embeddings.generate(article)

# Search:
hits =
  DefactoAI.SimilaritySearch.search("how to do X", limit: 5, embeddable_types: [:article])
# => [%{embeddable_type: "article", embeddable_id: "...", score: 0.91, chunks: [...]}, ...]
```

## Telemetry events

```
[:defacto_ai, :complete_structured, :start | :stop | :exception]
[:defacto_ai, :complete_chat,        :start | :stop | :exception]
[:defacto_ai, :embed,                 :start | :stop | :exception]
[:defacto_ai, :embed_batch,           :start | :stop | :exception]
[:defacto_ai, :similarity_search,     :start | :stop | :exception]
[:defacto_ai, :strategy, :rejected]   # emitted when a strategy is skipped
[:defacto_ai, :repair_loop, :retry]   # emitted on every corrective retry
```

`stream_chat/2` is intentionally not wrapped in a span — the span would
close before the stream is consumed.

## Testing

In `config/test.exs`:

```elixir
config :defacto_ai, :client, DefactoAI.Client.Stub
```

In your tests:

```elixir
import DefactoAI.Client.Stub, only: [expect: 2, expect_chat: 1, reset: 0]

setup do
  reset()
  :ok
end

test "..." do
  expect(MyApp.AI.QuizResponse, fn _msgs, _opts ->
    {:ok, %MyApp.AI.QuizResponse{title: "Test", questions: [...]}}
  end)

  assert {:ok, %MyApp.AI.QuizResponse{}} =
           MyApp.AI.generate_quiz("...")
end
```

Tests for embeddings can stub `Req` directly via `Req.Test` and pass the
test plug as the `:plug` opt.

## Development

```bash
mix deps.get
mix test
mix format
```

## Status

Pre-1.0; the public API is stabilising. The library is in production
use, but API breakage between minor versions is possible until a `1.0`
tag is cut.
