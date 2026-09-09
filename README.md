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
| `:stream_include_usage` | `false` | Send `stream_options: %{include_usage: true}` on streamed requests so the provider reports token usage. Off by default because providers that do not know the option may reject the request, and some providers report usage without it. See [Telemetry](#telemetry). |
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

## Telemetry

Every call to the LLM or the embeddings endpoint emits
[`:telemetry`](https://hexdocs.pm/telemetry) events, so a host can record
what each call cost and which part of the app made it. The library only
emits events. Where they go (a database table, metrics, logs) is up to the
handler you attach.

### Events

Calls that are wrapped in `:telemetry.span/3` emit `:start`, `:stop` and
`:exception` events. `stream_chat` emits the same three events, but by hand
(see below). Durations are in native time units.

| Event | Emitted by |
|---|---|
| `[:defacto_ai, :complete_structured, :start \| :stop \| :exception]` | `DefactoAI.Client.complete_structured/3` |
| `[:defacto_ai, :complete_chat, :start \| :stop \| :exception]` | `DefactoAI.Client.complete_chat/2` |
| `[:defacto_ai, :stream_chat, :start \| :stop \| :exception]` | `DefactoAI.Client.stream_chat/2` |
| `[:defacto_ai, :embed, :start \| :stop \| :exception]` | `DefactoAI.Embeddings.embed/2` |
| `[:defacto_ai, :embed_batch, :start \| :stop \| :exception]` | `DefactoAI.Embeddings.embed_batch/2` |
| `[:defacto_ai, :similarity_search, :start \| :stop \| :exception]` | `DefactoAI.SimilaritySearch.search/2` |
| `[:defacto_ai, :strategy, :rejected]` | a structured-output strategy was skipped |
| `[:defacto_ai, :repair_loop, :retry]` | a corrective retry was started |

### Measurements

| Event | Measurements |
|---|---|
| `:start` | `monotonic_time`, `system_time` |
| `:stop` | `duration`, `monotonic_time` |
| `:exception` | `duration`, `monotonic_time` |
| `[:defacto_ai, :strategy, :rejected]` | `count` (always `1`) |
| `[:defacto_ai, :repair_loop, :retry]` | `remaining` (retries left after this one) |

### Metadata

The `:stop` events of `complete_structured`, `complete_chat`, `stream_chat`,
`embed` and `embed_batch` all carry:

| Key | Value |
|---|---|
| `provider` | The resolved provider's `provider_type` as a string, or `nil` when the provider has none or could not be resolved. |
| `model` | The resolved provider's model as a string, or `nil` when no provider was resolved. |
| `role` | The `role:` the call was made with (`:llm` by default for chat, `:embedding` for embeddings). `nil` when only `provider:` was given. |
| `usage` | `%{input_tokens: integer \| nil, output_tokens: integer \| nil}`. Summed over every request the call made, including validation retries and strategy fallbacks. A value is `nil` when the provider did not report it. |
| `result` | `:ok` for an `{:ok, _}` result, `:error` for anything else. |
| `tags` | The map passed as `telemetry_tags:`, `%{}` by default. |

Some events add keys of their own:

  - `complete_structured` also has `schema` (the response schema module).
  - `embed_batch` also has `count` (number of texts).
  - `:start` events carry `role` and `tags`, plus `schema` or `count` where
    the `:stop` event has them. They do not carry `provider`, `model`,
    `usage` or `result`, because those are not known yet.
  - `:exception` events carry `kind`, `reason` and `stacktrace`. For the
    span-based events that is the start metadata plus those keys. For
    `stream_chat`, an exception while the stream is consumed carries the
    `:stop` metadata plus those keys. If `stream_chat/2` itself raises, the
    metadata is `role`, `tags`, `kind`, `reason` and `stacktrace`.

Three events do not carry the keys above. `similarity_search` has an empty
`:start` metadata and `count` (hits returned) on `:stop`.
`[:defacto_ai, :strategy, :rejected]` carries `strategy` and `reason`.
`[:defacto_ai, :repair_loop, :retry]` carries `schema`.

### Tagging calls

Pass `telemetry_tags:` to say which feature made a call:

```elixir
DefactoAI.Client.complete_chat(messages, telemetry_tags: %{feature: :my_feature})
```

All of `complete_structured/3`, `complete_chat/2`, `stream_chat/2`,
`Embeddings.embed/2` and `Embeddings.embed_batch/2` accept it. The option is
only used for telemetry. It is never sent to the provider and is not passed
on to the schema changeset.

### Where usage comes from

  - **`complete_chat`.** LangChain's token usage on the assistant message,
    summed over every request that returned one.
  - **`complete_structured`.** The same, including validation retries and
    strategy fallbacks.
  - **`embed`.** The response's `prompt_tokens`, reported as `input_tokens`.
    **`embed_batch`.** The same, summed over the texts. Embedding endpoints
    report no output tokens, so `output_tokens` is always `nil` there.
  - **`stream_chat`.** The `usage` object in the SSE chunks, when the provider
    sends one. Otherwise `input_tokens` and `output_tokens` stay `nil`.

Providers differ on streamed usage. OpenAI only sends it when asked with
`stream_options: %{include_usage: true}`. Some providers report it without
being asked: Heroku Managed Inference returns a `usage` object in the last
SSE chunk of a streamed chat completion with or without the option (verified
2026-11-13). The library reads usage from the stream whenever a provider
sends it, so for those providers it is safe to leave `:stream_include_usage`
unset.

Turn `:stream_include_usage` on only when the providers you stream from follow
OpenAI's opt-in behaviour. Providers that do not know the option may reject
the whole request, so the library does not send it unless you configure it:

```elixir
config :defacto_ai, stream_include_usage: true
```

This applies to `stream_chat/2` and, when `:chat_stream` is `true`,
to streamed structured calls. The key is global: turn it on only when
every provider you stream from accepts `stream_options`.

### `stream_chat` events

A stream is lazy, so there is no span around it. `:start` is emitted at the start of
`stream_chat/2`, before it returns. `:stop` is emitted when the stream
ends, when the consumer halts it early (for example with `Enum.take/2`) or
when the stream yields `{:error, reason}` and stops. `result` is `:error` if
the stream yielded an error element, and `duration` runs from the `:start`
event to the end of the stream. If enumerating the stream raises, an
`:exception` event is emitted instead of `:stop`. A stream that is never
enumerated never emits `:stop`.

If `stream_chat/2` itself returns `{:error, reason}`, `:stop` is emitted
before it returns, with `result: :error`. If it raises, `:exception` is
emitted instead of `:stop`.

### Example: record every call to the LLM or the embeddings endpoint

```elixir
defmodule MyApp.AI.UsageRecorder do
  def attach do
    events =
      for name <- [:complete_structured, :complete_chat, :stream_chat, :embed, :embed_batch] do
        [:defacto_ai, name, :stop]
      end

    :telemetry.attach_many("my-app-ai-usage", events, &__MODULE__.handle_event/4, nil)
  end

  def handle_event(event, %{duration: duration}, metadata, _config) do
    MyApp.Repo.insert!(%MyApp.AI.CallLog{
      call: event |> Enum.at(1) |> Atom.to_string(),
      provider: metadata.provider,
      model: metadata.model,
      feature: metadata.tags[:feature] && to_string(metadata.tags[:feature]),
      input_tokens: metadata.usage.input_tokens,
      output_tokens: metadata.usage.output_tokens,
      result: Atom.to_string(metadata.result),
      duration_ms: System.convert_time_unit(duration, :native, :millisecond)
    })
  end
end
```

Call `MyApp.AI.UsageRecorder.attach/0` once when your application starts.
Handlers run in the process that emits the event. That is the caller,
except for a stream's `:stop` and a mid-stream `:exception`, which run in the
process that consumes the stream. Keep handlers fast, or hand the row to a job queue.

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
