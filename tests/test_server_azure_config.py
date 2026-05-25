import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

SERVER_DIR = Path(__file__).resolve().parents[1] / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


def load_server_main(env):
    test_env = {"AUTH_DISABLED": "true", "JWT_SECRET": "test-secret", **env}
    with patch.dict(os.environ, test_env, clear=False):
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
    server_main = load_server_main(
        {
            "OPENAI_API_KEY": "fake-key",
            "ADMIN_API_KEY": "",
            "MEM0_DEFAULT_LLM_PROVIDER": "openai",
            "MEM0_DEFAULT_EMBEDDER_PROVIDER": "openai",
            "MEM0_DEFAULT_LLM_MODEL": "gpt-4.1-nano-2025-04-14",
            "MEM0_DEFAULT_EMBEDDER_MODEL": "text-embedding-3-small",
            "MEM0_DEFAULT_EMBEDDING_DIMS": "",
        }
    )

    config = server_main.DEFAULT_CONFIG

    assert config["llm"]["provider"] == "openai"
    assert config["llm"]["config"]["api_key"] == "fake-key"
    assert "azure_kwargs" not in config["llm"]["config"]
    assert config["embedder"]["provider"] == "openai"
    assert config["embedder"]["config"]["api_key"] == "fake-key"
    assert "azure_kwargs" not in config["embedder"]["config"]


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


def test_llm_patch_uses_max_completion_tokens_for_modern_gpt_models():
    server_main = load_server_main({"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": ""})
    server_main._apply_patches()

    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.llms.base import LLMBase

    class TestLLM(LLMBase):
        def generate_response(self, messages, tools=None, tool_choice="auto", **kwargs):
            return "ok"

    llm = TestLLM.__new__(TestLLM)
    llm.config = BaseLlmConfig(model="gpt-5.4-mini", max_tokens=123, temperature=0.2, top_p=0.1)

    params = llm._get_common_params()

    assert params["max_completion_tokens"] == 123
    assert "max_tokens" not in params
    assert params["temperature"] == 0.2
