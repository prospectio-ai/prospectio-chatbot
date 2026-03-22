"""
Tests for prospectio_chatbot/config.py - Settings validation via env vars.
"""
import pytest
from config import (
    ChainlitSettings,
    OllamaSettings,
    GeminiSettings,
    MistralSettings,
    PostgreSettings,
    OpenRouterSettings,
    MCPSettings,
)


class TestChainlitSettings:
    def test_loads_models_list_from_env(self, mock_env_config):
        settings = ChainlitSettings()
        assert settings.MODELS_LIST == ["Ollama/llama3", "Google/gemini-pro"]

    def test_loads_allowed_origins_from_env(self, mock_env_config):
        settings = ChainlitSettings()
        assert settings.ALLOWED_ORIGINS == ["http://localhost:3000"]

    def test_missing_models_list_raises(self, monkeypatch):
        monkeypatch.delenv("MODELS_LIST", raising=False)
        monkeypatch.setenv("ALLOWED_ORIGINS", '["http://localhost"]')
        with pytest.raises(Exception):
            ChainlitSettings()


class TestOllamaSettings:
    def test_loads_base_url(self, mock_env_config):
        settings = OllamaSettings()
        assert settings.OLLAMA_BASE_URL == "http://localhost:11434"


class TestGeminiSettings:
    def test_loads_api_key(self, mock_env_config):
        settings = GeminiSettings()
        assert settings.GOOGLE_API_KEY == "fake-google-key"


class TestMistralSettings:
    def test_loads_api_key(self, mock_env_config):
        settings = MistralSettings()
        assert settings.MISTRAL_API_KEY == "fake-mistral-key"


class TestPostgreSettings:
    def test_loads_connection_string(self, mock_env_config):
        settings = PostgreSettings()
        assert "postgresql://" in settings.POSTGRE_CONNECTION_STRING


class TestOpenRouterSettings:
    def test_loads_api_url_and_key(self, mock_env_config):
        settings = OpenRouterSettings()
        assert settings.OPEN_ROUTER_API_URL == "https://openrouter.ai/api/v1"
        assert settings.OPEN_ROUTER_API_KEY == "fake-openrouter-key"


class TestMCPSettings:
    def test_loads_empty_servers_list(self, mock_env_config):
        settings = MCPSettings()
        assert settings.MCP_SERVERS == []
