# Mem0 Self-Hosted Docker Runbook

## Purpose

This runbook documents the reliable self-hosted Mem0 setup used as the backend contract for a future Hermes `mem0-local` plugin.

## Services

- API: `http://localhost:8888`
- Dashboard: `http://localhost:3001`
- Postgres/pgvector: host `localhost:8432`, container `postgres:5432`

## First Start

```fish
cd server
cp .env.example .env
docker compose up -d --build
```

Fill `.env` before starting in production. Set `JWT_SECRET` unless `AUTH_DISABLED=true` is only being used for local development.

## Azure OpenAI Setup

Use deployment names from your Azure OpenAI resource:

```dotenv
MEM0_DEFAULT_LLM_PROVIDER=azure_openai
MEM0_DEFAULT_LLM_MODEL=gpt-5.4-mini
MEM0_DEFAULT_EMBEDDER_PROVIDER=azure_openai
MEM0_DEFAULT_EMBEDDER_MODEL=text-embedding-3-large
MEM0_DEFAULT_EMBEDDING_DIMS=1536
AZURE_OPENAI_API_KEY=replace-with-your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://replace-with-your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

`text-embedding-3-large` normally returns 3072 dimensions. This setup uses 1536 because the pgvector HNSW index cannot handle 3072-dimensional vectors in this environment.

Treat the embedding dimension as part of the persisted schema contract. Changing it later requires a deliberate migration and re-embedding existing memories.

## Restart Rule

Environment changes require a full Compose down/up:

```fish
cd server
docker compose down
docker compose up -d
```

Do not rely on uvicorn reload or a simple container restart for `.env` changes.

## Persistence Verification

Do not trust `POST /memories` alone. Verify the memory can be read back.

```fish
set api_key "replace-with-admin-or-user-api-key"
set agent "dimension-regression-probe"
curl -sS -X POST http://localhost:8888/memories \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $api_key" \
  -d '{"messages":[{"role":"user","content":"Rodger prefers Fish shell for manual commands."}],"agent_id":"'$agent'"}'

curl -sS "http://localhost:8888/memories?agent_id=$agent" \
  -H "X-API-Key: $api_key"
```

The GET response must include the newly added memory.

Check logs for dimension errors:

```fish
docker compose logs mem0 | grep "expected 1536 dimensions, not 3072"
```

No matching recent error should appear after the fix.

## Hermes Compatibility Contract

The future Hermes plugin should use:

- Provider name: `mem0-local`
- Base URL on the same host: `http://localhost:8888`
- Auth header: `X-API-Key`
- Identity: `agent_id=hermes-agent`, `user_id=null`
- Self-hosted routes: `/memories` and `/search`, not Mem0 Cloud `/v3/...` routes

The Hermes plugin itself is a separate follow-up project.

## Native 3072-Dimension Support

Native indexed `text-embedding-3-large` at 3072 dimensions is intentionally out of scope for this hardening project.

The recommended production default here is `text-embedding-3-large` with `MEM0_DEFAULT_EMBEDDING_DIMS=1536`. OpenAI and Azure OpenAI support this dimension reduction directly through the embeddings API, and it keeps the pgvector HNSW index usable.

A future project can evaluate `halfvec(3072)`, storing full `vector(3072)` with an indexed `halfvec` expression, or another vector backend. That project must include schema migration and re-embedding strategy.

## Troubleshooting

- Dashboard build fails on Node 20: rebuild with the included Node 22 Dockerfile.
- New writes return `200 OK` but do not appear in GET results: check logs for pgvector dimension errors.
- `expected 1536 dimensions, not 3072`: confirm `MEM0_DEFAULT_EMBEDDING_DIMS=1536`, full down/up was run, and Azure batch embeddings pass dimensions.
- Search ranking looks inverted: confirm `mem0/vector_stores/pgvector.py` converts pgvector cosine distance to similarity.
