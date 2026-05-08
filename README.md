# DefactoAI

Shared LLM, embeddings and similarity-search building blocks for Defacto apps.

This package extracts the universal pieces of [Detroit](../detroit)'s in-house
AI client (introduced in PR #17919) into a stand-alone Mix dependency so that
they can be reused across Detroit, [Quizmass](../quizmass), and other Defacto
Elixir/Phoenix apps without copy-paste.

## What's in the box

| Concern | Module |
|---|---|
| Structured / chat / streaming completions | `DefactoAI.Client` |
| Strategy fallback (tool call → JSON mode → text repair) | `DefactoAI.Strategy` |
| LLM-output JSON repair pipeline | `DefactoAI.JSON` |
| Validation-failure repair loop | `DefactoAI.RepairLoop` |
| LangChain `ChatOpenAI` adapter | `DefactoAI.LangChainAdapter` |
| Provider config contract | `DefactoAI.Provider` (protocol) |
| Embeddings HTTP client + store | `DefactoAI.Embeddings`, `DefactoAI.Embedding` |
| Vector similarity search | `DefactoAI.SimilaritySearch` |
| Test stub | `DefactoAI.Client.Stub` |

## Status

This is an **early-stage path-dep** package. It is not yet pushed to GitHub
or Hex. Until the API stabilises, host apps consume it via:

```elixir
# in mix.exs
{:defacto_ai, path: "../defacto_ai"}
```

## Quick start

See the host application's docs (Detroit, Quizmass) for the integration
glue. At minimum a host app must:

1. Add the dep above.
2. Configure the package in `config/config.exs`:
   ```elixir
   config :defacto_ai,
     repo: MyApp.Repo,
     provider_resolver: &MyApp.AI.resolve_provider/1
   ```
3. Implement the `DefactoAI.Provider` protocol on the app's own provider
   schema.
4. Run `mix defacto_ai.gen.migration` to install the embeddings table
   (only if using the embeddings/similarity-search modules).

## Status checklist

- [ ] Strategy / JSON / RepairLoop / LangChainAdapter
- [ ] Provider protocol
- [ ] Client behaviour + LangChain implementation
- [ ] Client.Stub
- [ ] PromptRenderer
- [ ] Embeddings: Chunker, RateLimiter, schema, HTTP client, context
- [ ] SimilaritySearch
- [ ] `mix defacto_ai.gen.migration`
- [ ] Telemetry events
- [ ] README usage docs

## Development

```bash
mix deps.get
mix test
mix format
```
