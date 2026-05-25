# Self-Hosted Mem0 Hermes-Ready Server Design

## Goal

Make the self-hosted Mem0 Docker/server setup reliable and professional so it can be used as the stable backend for a future update-safe Hermes Agent memory plugin.

This design covers the server, Docker Compose, dashboard container, Azure OpenAI configuration, pgvector persistence, verification, and documentation. The Hermes plugin itself is intentionally deferred to a separate follow-up project, but this work defines the server-side compatibility contract that plugin will depend on.

## Current Problems

The session log in `opencode-session-mem0-config.md` shows that the working deployment required several fixes that are not yet present in this fork:

- The dashboard Dockerfile uses `node:20-alpine`, but the lockfile/pnpm 11 path requires Node 22.
- Dashboard dependency installation needs a pnpm build-approval-safe pattern for packages with build scripts.
- The Compose dashboard host port currently defaults to `3000`, which commonly conflicts with other local services.
- The self-hosted server only lists OpenAI, Anthropic, and Gemini as bundled providers; Azure OpenAI is not first-class in the server defaults.
- `.env.example` lacks Azure OpenAI provider, endpoint, model, deployment, API version, and embedding-dimension settings.
- Azure config must use nested `azure_kwargs`; top-level Azure fields break the mem0 config classes.
- Newer GPT models can reject `max_tokens` and need `max_completion_tokens`.
- `text-embedding-3-large` returns 3072-dimensional embeddings by default, while the working pgvector/HNSW setup must stay at `vector(1536)` because HNSW cannot index 3072 dimensions in this setup.
- `AzureOpenAIEmbedding.embed_batch()` ignores `embedding_dims`, so add/insert paths can produce 3072-dimensional vectors even when search uses 1536-dimensional vectors.
- pgvector `<=>` returns cosine distance, but downstream ranking expects a similarity score where higher is better.
- `POST /memories` can return `200 OK` while inserts fail internally, so verification must read the memory back and inspect logs.

## Scope

Project 1 fixes the self-hosted Mem0 server and Docker setup.

Included:

- Production-ready Dockerfile/Compose behavior for the self-hosted API, Postgres/pgvector, and dashboard.
- Dashboard Dockerfile fixes for Node 22 and pnpm 11 dependency installation.
- Env-driven OpenAI/Azure OpenAI provider selection for the self-hosted server.
- Azure OpenAI default config builders that use nested `azure_kwargs`.
- Azure embedding dimension handling for both single and batch embedding paths.
- Modern LLM token parameter compatibility.
- pgvector distance-to-similarity score correction.
- `.env.example` updates with safe placeholders and documented defaults.
- A runbook with startup, restart, verification, troubleshooting, and Hermes compatibility notes.
- Tests or focused verification checks for the compatibility fixes where practical.

Excluded:

- Building or packaging the Hermes `mem0-local` plugin.
- Changing Hermes user config files.
- Automating remote `pi5-02` deployment.
- Automating Caddy configuration.
- Changing upstream mem0 public APIs unless required for the server fixes.

## Design Approach

Use a server-first design with a Hermes compatibility contract.

The self-hosted server must be reliable before the Hermes plugin is hardened. The fixes should land in this fork as maintainable server behavior, not as one-off edits to a deployed host. Hermes plugin work becomes Project 2 after this server foundation is tested.

Where the underlying mem0 SDK has provider-specific compatibility gaps, prefer small and explicit server-local compatibility code in `server/main.py` only when that is the least disruptive way to keep self-hosted deployments stable. Avoid hidden direct edits to upstream SDK files that are likely to be overwritten by package updates.

## Server Configuration

Add env-driven defaults for providers and models:

- `MEM0_DEFAULT_LLM_PROVIDER`, default `openai`.
- `MEM0_DEFAULT_LLM_MODEL`, default existing OpenAI model.
- `MEM0_DEFAULT_EMBEDDER_PROVIDER`, default `openai`.
- `MEM0_DEFAULT_EMBEDDER_MODEL`, default existing OpenAI embedding model.
- `MEM0_DEFAULT_EMBEDDING_DIMS`, optional integer.
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`.
- Optional deployment-specific aliases if the SDK expects deployment names through `azure_kwargs.azure_deployment`.

For Azure OpenAI:

- LLM config must set `provider: azure_openai` or `azure_openai_structured` only when selected.
- Embedder config must set `provider: azure_openai` only when selected.
- Azure settings must be nested under `config.azure_kwargs`.
- `embedding_dims` must be included when configured, especially for `text-embedding-3-large` with pgvector/HNSW.

The provider allowlist should include Azure providers that are actually bundled in the image.

## Docker And Dashboard

The dashboard Dockerfile should use Node 22 because the current Next/pnpm lockfile path requires a Node version compatible with pnpm 11.

Dependency installation should use a pnpm 11-compatible build approval strategy rather than `--ignore-scripts`. The working session used the pattern:

```sh
pnpm i || (pnpm approve-builds --all && pnpm i)
```

The Compose file should avoid a default dashboard host conflict by mapping the dashboard to `3001:3000`, while the container continues to listen on port `3000`.

The API should continue to expose host port `8888` for local/Hermes access. Postgres should keep a named volume for persistence.

## Embedding And Vector Store Behavior

The default Azure Hermes-ready setup should use:

- LLM: `gpt-5.4-mini` when configured by env.
- Embedder: `text-embedding-3-large` when configured by env.
- Embedding dimensions: `1536` for pgvector/HNSW compatibility.

Rationale:

- `text-embedding-3-large` naturally returns 3072 dimensions.
- This pgvector/HNSW setup cannot index vectors above 2000 dimensions.
- OpenAI/Azure supports dimension reduction for `text-embedding-3-large`.
- Keeping `vector(1536)` avoids a fragile DB/index migration and preserves HNSW.

Both single and batch Azure embedding calls must pass the configured `dimensions` value. This specifically prevents the known failure mode where search vectors are 1536 but add/insert vectors are 3072.

pgvector search results should convert cosine distance to similarity before returning `OutputData.score`:

```python
score = max(0.0, min(1.0, 1.0 - distance))
```

This matches the downstream ranking expectation that higher scores are better.

## Modern LLM Token Parameters

The server should handle modern GPT deployments that reject `max_tokens` and require `max_completion_tokens`.

The compatibility logic should be narrow and model-aware. It should preserve existing behavior for models that still accept `max_tokens`.

## Hermes Compatibility Contract

The future Hermes `mem0-local` plugin will depend on these server guarantees:

- The API accepts `X-API-Key` authentication for self-hosted access.
- The API exposes self-hosted routes such as `/memories` and `/search`, not Mem0 Cloud `/v3/...` routes.
- `POST /memories` with `agent_id` persists records that are retrievable via `GET /memories?agent_id=<agent>`.
- Search works against the same `agent_id` scope.
- The recommended Hermes identity is `agent_id: hermes-agent` and `user_id: null`.
- Hermes running on the same host should use `http://localhost:8888`, not the public Caddy URL.
- Public dashboard/browser access can use HTTPS/Caddy separately, but that is not part of the plugin runtime contract.

The server verification must not trust `POST /memories` alone. A valid smoke test must add a unique fact, fetch it back by `agent_id`, and confirm server logs do not contain `expected 1536 dimensions, not 3072`.

## Testing And Verification

Verification should cover:

- Dashboard image builds with Node 22 and pnpm 11.
- Compose stack starts with Postgres healthy, API running, and dashboard healthy.
- `/configure/providers` includes Azure providers when bundled.
- Azure config generation uses nested `azure_kwargs`.
- Azure single and batch embeddings honor `embedding_dims`.
- A memory added through `POST /memories` is returned by `GET /memories?agent_id=<probe-agent>`.
- pgvector result scores behave as similarity values.
- A full `docker compose down && docker compose up -d` reloads changed env values.

Automated unit tests are preferred for pure config and patch behavior. Docker/API smoke verification should be documented and run manually or through a targeted script if practical.

## Documentation

Add or update a root-level self-hosted runbook. It should include:

- Required env vars.
- Azure OpenAI example config using placeholders only.
- Start and rebuild commands.
- The full restart rule for env/config changes.
- Persistence verification commands.
- Troubleshooting for 1536/3072 embedding mismatch.
- Dashboard port notes.
- Hermes compatibility notes.
- A clear statement that the Hermes plugin itself is a separate follow-up project.

## Follow-Up Project

Project 2 should harden the Hermes `mem0-local` plugin after Project 1 passes.

That project should cover:

- Update-safe plugin location and naming.
- Self-hosted endpoint client using `X-API-Key`.
- `agent_id`-based add/search/list/delete behavior.
- Install/update docs.
- Smoke tests against the self-hosted server contract.

## Risks

- Runtime monkey patches can hide upstream behavior changes. Keep them small, documented, and covered by tests or smoke checks.
- Azure deployment names can differ from model names. The env docs must be explicit about deployment/model fields.
- Existing Postgres volumes may already contain tables with different vector dimensions. The runbook must explain when a volume reset or migration is required.
- Docker env changes do not reload with a simple uvicorn reload or container restart in all cases. The supported rule is full Compose down/up.
