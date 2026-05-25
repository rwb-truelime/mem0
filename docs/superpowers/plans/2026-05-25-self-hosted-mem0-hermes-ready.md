# Self-Hosted Mem0 Hermes-Ready Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the self-hosted Mem0 Docker/server setup reliable for Azure OpenAI, pgvector persistence, and a future update-safe Hermes Agent plugin.

**Architecture:** Keep the self-hosted server as the compatibility boundary. Add small env-driven config builders and explicit runtime compatibility patches in `server/main.py`, update Docker/dashboard assets, and document the operational contract Hermes will rely on. Do not implement the Hermes plugin in this phase.

**Tech Stack:** Python 3.12, FastAPI, pytest, mem0 Python SDK, PostgreSQL/pgvector, Docker Compose, Next.js dashboard, Node 22, pnpm 11.

---

## File Structure

- Modify `server/main.py`: Add env-driven provider config helpers, Azure provider allowlist entries, runtime compatibility patches, and call patches before server state initialization.
- Modify `server/requirements.txt`: Add `azure-identity` because Azure OpenAI providers import it for credential fallback.
- Modify `server/.env.example`: Document OpenAI/Azure provider selection, Azure nested-config inputs, embedding dimensions, ports, auth, and restart expectations.
- Modify `server/docker-compose.yaml`: Make Compose defaults match the reliable local/Hermes stack: API on `8888`, dashboard host on `3001`, persistent Postgres volume, and dashboard URL consistency.
- Modify `server/dashboard/Dockerfile`: Move dashboard build to Node 22 and use a pnpm 11-safe install path.
- Create `tests/test_server_azure_config.py`: Pure unit tests for provider lists, default config generation, and runtime patch behavior with mocked clients.
- Create or update `MEM0-SELF-HOSTED-RUNBOOK.md`: Root-level runbook for Docker, Azure, persistence verification, troubleshooting, and Hermes compatibility contract.
- Keep Azure embedding and LLM compatibility patches server-local. Implement pgvector distance-to-similarity as a permanent SDK fix in `mem0/vector_stores/pgvector.py` because this fork should carry that bug fix directly.

## Pre-Execution Requirements

- [ ] **Step 1: Confirm the branch**

Run: `git branch --show-current`

Expected: `fix/self-hosted-mem0-hermes-ready`

- [ ] **Step 2: Inspect worktree status**

Run: `git status --short`

Expected: Existing uncommitted files may include the design and plan docs. Do not revert unrelated user changes.

- [ ] **Step 3: Run GitNexus impact before editing server symbols**

Run these GitNexus checks before editing `server/main.py` symbols:

```text
gitnexus_impact({target: "initialize_state", direction: "upstream", repo: "mem0"})
gitnexus_impact({target: "AzureOpenAIEmbedding", direction: "upstream", repo: "mem0", relationTypes: ["CALLS", "IMPORTS", "HAS_METHOD"]})
gitnexus_impact({target: "PGVector", direction: "upstream", repo: "mem0", relationTypes: ["CALLS", "IMPORTS", "HAS_METHOD"]})
```

Expected: Record risk level and direct callers in the task notes. If HIGH or CRITICAL, stop and warn before editing.

---

### Task 1: Add Azure Config Builders And Provider Defaults

**Files:**
- Modify: `server/main.py:55-132`
- Test: `tests/test_server_azure_config.py`

- [ ] **Step 1: Write failing tests for env-driven Azure default config**

Create `tests/test_server_azure_config.py` with this initial content:

```python
import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")


def load_server_main(env):
    with patch.dict(os.environ, env, clear=False):
        with patch("mem0.Memory.from_config", return_value=MagicMock()):
            import server.main as server_main
            return importlib.reload(server_main)


def test_azure_providers_are_bundled():
    server_main = load_server_main({"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": ""})

    assert "azure_openai" in server_main.BUNDLED_LLM_PROVIDERS
    assert "azure_openai_structured" in server_main.BUNDLED_LLM_PROVIDERS
    assert "azure_openai" in server_main.BUNDLED_EMBEDDER_PROVIDERS


def test_default_config_uses_nested_azure_kwargs_for_azure_openai():
    server_main = load_server_main(
        {
            "ADMIN_API_KEY": "",
            "MEM0_DEFAULT_LLM_PROVIDER": "azure_openai",
            "MEM0_DEFAULT_LLM_MODEL": "gpt-5.4-mini",
            "MEM0_DEFAULT_EMBEDDER_PROVIDER": "azure_openai",
            "MEM0_DEFAULT_EMBEDDER_MODEL": "text-embedding-3-large",
            "MEM0_DEFAULT_EMBEDDING_DIMS": "1536",
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-08-01-preview",
        }
    )

    config = server_main.DEFAULT_CONFIG

    assert config["llm"]["provider"] == "azure_openai"
    assert config["llm"]["config"]["model"] == "gpt-5.4-mini"
    assert config["llm"]["config"]["azure_kwargs"] == {
        "api_key": "azure-key",
        "azure_deployment": "gpt-5.4-mini",
        "azure_endpoint": "https://example.openai.azure.com",
        "api_version": "2024-08-01-preview",
    }

    assert config["embedder"]["provider"] == "azure_openai"
    assert config["embedder"]["config"]["model"] == "text-embedding-3-large"
    assert config["embedder"]["config"]["embedding_dims"] == 1536
    assert config["embedder"]["config"]["azure_kwargs"] == {
        "api_key": "azure-key",
        "azure_deployment": "text-embedding-3-large",
        "azure_endpoint": "https://example.openai.azure.com",
        "api_version": "2024-08-01-preview",
    }


def test_openai_defaults_remain_openai_when_provider_env_is_omitted():
    server_main = load_server_main({"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": ""})

    config = server_main.DEFAULT_CONFIG


    assert config["llm"]["provider"] == "openai"
    assert config["llm"]["config"]["api_key"] == "fake-key"
    assert "azure_kwargs" not in config["llm"]["config"]
    assert config["embedder"]["provider"] == "openai"
    assert config["embedder"]["config"]["api_key"] == "fake-key"
    assert "azure_kwargs" not in config["embedder"]["config"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_server_azure_config.py -q`

Expected: FAIL because Azure providers/config builders do not exist yet.

- [ ] **Step 3: Implement minimal provider config helpers**

Modify `server/main.py` around the provider constants and default config setup to include:

```python
BUNDLED_LLM_PROVIDERS = ("openai", "anthropic", "gemini", "azure_openai", "azure_openai_structured")
BUNDLED_EMBEDDER_PROVIDERS = ("openai", "gemini", "azure_openai")
```

Replace the current `OPENAI_API_KEY`, `DEFAULT_LLM_MODEL`, `DEFAULT_EMBEDDER_MODEL`, and `DEFAULT_CONFIG` block with helper functions equivalent to:

```python
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")
DEFAULT_LLM_PROVIDER = os.environ.get("MEM0_DEFAULT_LLM_PROVIDER", "openai")
DEFAULT_LLM_MODEL = os.environ.get("MEM0_DEFAULT_LLM_MODEL", "gpt-4.1-nano-2025-04-14")
DEFAULT_EMBEDDER_PROVIDER = os.environ.get("MEM0_DEFAULT_EMBEDDER_PROVIDER", "openai")
DEFAULT_EMBEDDER_MODEL = os.environ.get("MEM0_DEFAULT_EMBEDDER_MODEL", "text-embedding-3-small")


def _optional_int(value: str | None) -> int | None:
    return int(value) if value else None


DEFAULT_EMBEDDING_DIMS = _optional_int(os.environ.get("MEM0_DEFAULT_EMBEDDING_DIMS"))


def _azure_kwargs(model: str) -> Dict[str, Any]:
    return {
        "api_key": AZURE_OPENAI_API_KEY,
        "azure_deployment": model,
        "azure_endpoint": AZURE_OPENAI_ENDPOINT,
        "api_version": AZURE_OPENAI_API_VERSION,
    }


def _build_llm_config() -> Dict[str, Any]:
    config: Dict[str, Any] = {"temperature": 0.2, "model": DEFAULT_LLM_MODEL}
    if DEFAULT_LLM_PROVIDER.startswith("azure_openai"):
        config["azure_kwargs"] = _azure_kwargs(DEFAULT_LLM_MODEL)
    else:
        config["api_key"] = OPENAI_API_KEY
    return {"provider": DEFAULT_LLM_PROVIDER, "config": config}


def _build_embedder_config() -> Dict[str, Any]:
    config: Dict[str, Any] = {"model": DEFAULT_EMBEDDER_MODEL}
    if DEFAULT_EMBEDDING_DIMS is not None:
        config["embedding_dims"] = DEFAULT_EMBEDDING_DIMS
    if DEFAULT_EMBEDDER_PROVIDER == "azure_openai":
        config["azure_kwargs"] = _azure_kwargs(DEFAULT_EMBEDDER_MODEL)
    else:
        config["api_key"] = OPENAI_API_KEY
    return {"provider": DEFAULT_EMBEDDER_PROVIDER, "config": config}
```

Then set `DEFAULT_CONFIG["llm"] = _build_llm_config()` and `DEFAULT_CONFIG["embedder"] = _build_embedder_config()`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_server_azure_config.py -q`

Expected: PASS for the three config tests.

- [ ] **Step 5: Run existing server tests**

Run: `pytest tests/test_server_auth.py tests/test_server_params.py -q`

Expected: PASS. If existing environment lacks optional dependencies, record the exact missing dependency and continue only after deciding whether to install or defer.

- [ ] **Step 6: Commit**

Run:

```bash
git add server/main.py tests/test_server_azure_config.py
git commit -m "fix(server): add azure openai default config"
```

Expected: Commit succeeds on branch `fix/self-hosted-mem0-hermes-ready`.

---

### Task 2: Add Runtime Compatibility Patches

**Files:**
- Modify: `server/main.py`
- Test: `tests/test_server_azure_config.py`

- [ ] **Step 1: Run GitNexus impact before editing runtime patch targets**

Run:

```text
gitnexus_impact({target: "AzureOpenAIEmbedding", direction: "upstream", repo: "mem0", relationTypes: ["CALLS", "IMPORTS", "HAS_METHOD"]})
gitnexus_impact({target: "PGVector", direction: "upstream", repo: "mem0", relationTypes: ["CALLS", "IMPORTS", "HAS_METHOD"]})
gitnexus_impact({target: "LLMBase", direction: "upstream", repo: "mem0", relationTypes: ["CALLS", "IMPORTS", "HAS_METHOD"]})
```

Expected: Record risks. Stop and warn if HIGH or CRITICAL.

- [ ] **Step 2: Add failing tests for patches**

Append these tests to `tests/test_server_azure_config.py`:

```python
from types import SimpleNamespace


def test_azure_embedding_patch_passes_dimensions_to_single_and_batch_calls():
    server_main = load_server_main({"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": ""})
    server_main._apply_patches()

    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.azure_openai import AzureOpenAIEmbedding

    class FakeEmbeddings:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            data = [SimpleNamespace(index=i, embedding=[float(i)] * 1536) for i, _ in enumerate(kwargs["input"])]
            return SimpleNamespace(data=data)

    embedder = AzureOpenAIEmbedding.__new__(AzureOpenAIEmbedding)
    embedder.config = BaseEmbedderConfig(model="text-embedding-3-large", embedding_dims=1536, azure_kwargs={})
    fake_embeddings = FakeEmbeddings()
    embedder.client = SimpleNamespace(embeddings=fake_embeddings)

    embedder.embed("hello")
    embedder.embed_batch(["first", "second"])

    assert fake_embeddings.calls[0]["dimensions"] == 1536
    assert fake_embeddings.calls[1]["dimensions"] == 1536
    assert fake_embeddings.calls[1]["input"] == ["first", "second"]


Update the existing `tests/vector_stores/test_pgvector.py` search assertions so raw pgvector distances `0.1` and `0.2` are expected as similarity scores `0.9` and `0.8`.


def test_llm_patch_uses_max_completion_tokens_for_modern_gpt_models():
    server_main = load_server_main({"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": ""})
    server_main._apply_patches()

    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.llms.base import LLMBase

    llm = LLMBase.__new__(LLMBase)
    llm.config = BaseLlmConfig(model="gpt-5.4-mini", max_tokens=123, temperature=0.2, top_p=1.0)

    params = llm._get_common_params()

    assert params["max_completion_tokens"] == 123
    assert "max_tokens" not in params
    assert params["temperature"] == 0.2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_server_azure_config.py -q`

Expected: FAIL because `_apply_patches()` does not exist yet or does not patch all three behaviors.

- [ ] **Step 4: Implement `_apply_patches()` and call it before `initialize_state`**

Add a function in `server/main.py` before `set_session_factory(SessionLocal)`:

```python
_PATCHES_APPLIED = False


def _model_uses_max_completion_tokens(model: str | None) -> bool:
    if not model:
        return False
    base_model = model.lower().rsplit("/", 1)[-1]
    return base_model.startswith(("gpt-4.1", "gpt-5"))


def _apply_patches() -> None:
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    from mem0.embeddings.azure_openai import AzureOpenAIEmbedding
    from mem0.llms.base import LLMBase

    original_embed = AzureOpenAIEmbedding.embed
    original_embed_batch = AzureOpenAIEmbedding.embed_batch
    original_get_common_params = LLMBase._get_common_params

    def patched_embed(self, text, memory_action=None):
        dimensions = getattr(self.config, "embedding_dims", None)
        if dimensions is None:
            return original_embed(self, text, memory_action)
        text = text.replace("\n", " ")
        return self.client.embeddings.create(
            input=[text],
            model=self.config.model,
            dimensions=dimensions,
        ).data[0].embedding

    def patched_embed_batch(self, texts, memory_action="add"):
        dimensions = getattr(self.config, "embedding_dims", None)
        if dimensions is None:
            return original_embed_batch(self, texts, memory_action)
        max_batch = 100
        texts = [text.replace("\n", " ") for text in texts]
        all_embeddings = []
        for i in range(0, len(texts), max_batch):
            chunk = texts[i : i + max_batch]
            response = self.client.embeddings.create(
                input=chunk,
                model=self.config.model,
                dimensions=dimensions,
            )
            all_embeddings.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return all_embeddings

    def patched_get_common_params(self, **kwargs):
        params = original_get_common_params(self, **kwargs)
        if _model_uses_max_completion_tokens(getattr(self.config, "model", None)) and "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
        return params

    AzureOpenAIEmbedding.embed = patched_embed
    AzureOpenAIEmbedding.embed_batch = patched_embed_batch
    LLMBase._get_common_params = patched_get_common_params
    _PATCHES_APPLIED = True
```

Then call:

```python
_apply_patches()
set_session_factory(SessionLocal)
initialize_state(DEFAULT_CONFIG)
```

- [ ] **Step 5: Run focused patch tests**

Run: `pytest tests/test_server_azure_config.py -q`

Expected: PASS.

- [ ] **Step 6: Run existing server tests**

Run: `pytest tests/test_server_auth.py tests/test_server_params.py tests/test_server_azure_config.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add server/main.py tests/test_server_azure_config.py
git commit -m "fix(server): patch azure embeddings and modern tokens"
```

Expected: Commit succeeds.

---

### Task 3: Update Docker, Dashboard, And Runtime Dependencies

**Files:**
- Modify: `server/dashboard/Dockerfile`
- Modify: `server/docker-compose.yaml`
- Modify: `server/requirements.txt`

- [ ] **Step 1: Update dashboard Dockerfile base and pnpm install**

Modify `server/dashboard/Dockerfile`:

```dockerfile
FROM node:22-alpine AS base
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Dependencies
FROM base AS deps
COPY package.json yarn.lock* package-lock.json* pnpm-lock.yaml* ./
RUN \
  if [ -f yarn.lock ]; then yarn --frozen-lockfile --network-timeout 600000; \
  elif [ -f package-lock.json ]; then npm ci; \
  elif [ -f pnpm-lock.yaml ]; then corepack enable pnpm && pnpm i || (pnpm approve-builds --all && pnpm i); \
  else npm install; \
  fi
```

Leave the builder and runner stages unchanged unless the build fails for a specific reason.

- [ ] **Step 2: Add Azure dependency**

Add this line to `server/requirements.txt` near the provider dependencies:

```text
azure-identity>=1.17,<2.0
```

- [ ] **Step 3: Update Compose dashboard host port and URL**

Modify `server/docker-compose.yaml`:

```yaml
    ports:
      - "3001:3000"
```

and set the API service dashboard URL to:

```yaml
      - DASHBOARD_URL=http://localhost:3001
```

Keep `NEXT_PUBLIC_API_URL=http://localhost:8888` and `API_INTERNAL_URL=http://mem0:8000`.

- [ ] **Step 4: Build the dashboard image**

Run: `docker compose build mem0-dashboard`

Working directory: `server`

Expected: Build completes with Node 22 and pnpm install succeeds. If Docker is unavailable, record the exact Docker error and run static verification in Step 5.

- [ ] **Step 5: Static verification if Docker is unavailable**

Run: `git diff -- server/dashboard/Dockerfile server/docker-compose.yaml server/requirements.txt`

Expected: Diff shows only Node 22, pnpm approval retry, dashboard port `3001:3000`, dashboard URL `http://localhost:3001`, and `azure-identity`.

- [ ] **Step 6: Commit**

Run:

```bash
git add server/dashboard/Dockerfile server/docker-compose.yaml server/requirements.txt
git commit -m "fix(server): harden docker compose defaults"
```

Expected: Commit succeeds.

---

### Task 4: Update Environment Example And Runbook

**Files:**
- Modify: `server/.env.example`
- Create: `MEM0-SELF-HOSTED-RUNBOOK.md`

- [ ] **Step 1: Update `.env.example`**

Replace `server/.env.example` content with a documented version containing:

```dotenv
# OpenAI fallback/default provider credentials.
OPENAI_API_KEY=

# Optional: other bundled LLM/embedder providers.
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# PostgreSQL/pgvector. Defaults in docker-compose use postgres/postgres/postgres.
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_COLLECTION_NAME=memories

# Self-hosted auth.
ADMIN_API_KEY=
JWT_SECRET=
AUTH_DISABLED=false
DASHBOARD_URL=http://localhost:3001
APP_DB_NAME=mem0_app

# Provider selection. Use openai defaults or switch both to azure_openai.
MEM0_DEFAULT_LLM_PROVIDER=openai
MEM0_DEFAULT_LLM_MODEL=gpt-4.1-nano-2025-04-14
MEM0_DEFAULT_EMBEDDER_PROVIDER=openai
MEM0_DEFAULT_EMBEDDER_MODEL=text-embedding-3-small

# Azure OpenAI example. Use deployment names that exist in your Azure resource.
# For Hermes-ready Azure setup, use:
# MEM0_DEFAULT_LLM_PROVIDER=azure_openai
# MEM0_DEFAULT_LLM_MODEL=gpt-5.4-mini
# MEM0_DEFAULT_EMBEDDER_PROVIDER=azure_openai
# MEM0_DEFAULT_EMBEDDER_MODEL=text-embedding-3-large
# MEM0_DEFAULT_EMBEDDING_DIMS=1536
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=2024-08-01-preview
MEM0_DEFAULT_EMBEDDING_DIMS=

# Anonymous telemetry. Sends a single onboarding event per install.
MEM0_TELEMETRY=true

# Days of request_logs history to keep when `make prune-logs` runs.
REQUEST_LOG_RETENTION_DAYS=30
```

- [ ] **Step 2: Create root runbook**

Create `MEM0-SELF-HOSTED-RUNBOOK.md` with sections:

```markdown
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

`text-embedding-3-large` normally returns 3072 dimensions. This setup uses `1536` because the pgvector HNSW index cannot handle 3072-dimensional vectors in this environment.

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

## Troubleshooting

- Dashboard build fails on Node 20: rebuild with the included Node 22 Dockerfile.
- New writes return `200 OK` but do not appear in GET results: check logs for pgvector dimension errors.
- `expected 1536 dimensions, not 3072`: confirm `MEM0_DEFAULT_EMBEDDING_DIMS=1536`, full down/up was run, and Azure batch embeddings pass dimensions.
- Search ranking looks inverted: confirm `mem0/vector_stores/pgvector.py` converts pgvector cosine distance to similarity.
```

- [ ] **Step 3: Verify docs contain no secrets**

Run: `grep -R "e899c1ce\|6d4cd866\|m0sk_" MEM0-SELF-HOSTED-RUNBOOK.md server/.env.example docs/superpowers -n`

Expected: No matches.

- [ ] **Step 4: Commit**

Run:

```bash
git add server/.env.example MEM0-SELF-HOSTED-RUNBOOK.md
git commit -m "docs(server): add self hosted runbook"
```

Expected: Commit succeeds.

---

### Task 5: End-To-End Verification And Change Detection

**Files:**
- No source edits unless verification exposes a defect.

- [ ] **Step 1: Run Python focused tests**

Run: `pytest tests/test_server_azure_config.py tests/test_server_auth.py tests/test_server_params.py -q`

Expected: PASS. If dependency setup is missing, record the exact missing package and run the relevant command from the repository setup docs before retrying.

- [ ] **Step 2: Run lint on touched Python files**

Run: `ruff check server/main.py tests/test_server_azure_config.py`

Expected: PASS.

- [ ] **Step 3: Build dashboard image**

Run: `docker compose build mem0-dashboard`

Working directory: `server`

Expected: PASS. If Docker is unavailable, record that Docker verification was not run and include the exact daemon error.

- [ ] **Step 4: Optional Compose smoke test with local secrets**

Only run if `.env` has valid local credentials.

Run:

```fish
cd server
docker compose down
docker compose up -d --build
```

Then run the persistence verification commands from `MEM0-SELF-HOSTED-RUNBOOK.md`.

Expected: Added memory is returned by `GET /memories?agent_id=dimension-regression-probe`, and logs do not show `expected 1536 dimensions, not 3072`.

- [ ] **Step 5: Run GitNexus change detection**

Run:

```text
gitnexus_detect_changes({scope: "all", repo: "mem0"})
```

Expected: Changed symbols and flows match server Docker/Azure/persistence scope.

- [ ] **Step 6: Review final diff**

Run: `git diff HEAD~4..HEAD --stat && git status --short`

Expected: Only intended files changed; working tree clean except intentionally uncommitted local files.

---

## Plan Self-Review Checklist

- Spec coverage: Tasks cover Docker/dashboard, env-driven Azure config, Azure embedding dimensions, modern token params, pgvector scoring, `.env.example`, runbook, and Hermes server contract.
- Scope control: Hermes plugin implementation is explicitly deferred.
- TDD: Tasks 1 and 2 start with failing tests before implementation.
- Verification: Focused tests, existing server tests, ruff, dashboard build, optional Compose smoke, and GitNexus change detection are included.
- Branching: Pre-execution requires branch `fix/self-hosted-mem0-hermes-ready` before commits.
