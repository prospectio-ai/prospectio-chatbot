"""
Mock fixtures for external dependencies: LLM clients, MCP servers, Chainlit, etc.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.fixture
def mock_env_config(monkeypatch):
    """Set all required environment variables for config classes."""
    monkeypatch.setenv("MODELS_LIST", '["Ollama/llama3", "Google/gemini-pro"]')
    monkeypatch.setenv("ALLOWED_ORIGINS", '["http://localhost:3000"]')
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-mistral-key")
    monkeypatch.setenv("POSTGRE_CONNECTION_STRING", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("OPEN_ROUTER_API_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "fake-openrouter-key")
    monkeypatch.setenv("MCP_SERVERS", '[]')
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "test-secret-key-for-jwt")


@pytest.fixture
def mock_llm_client():
    """Return a mock LLM client that behaves like a BaseChatModel."""
    client = MagicMock()
    client.bind_tools = MagicMock(return_value=client)
    client.ainvoke = AsyncMock(return_value=MagicMock(content="mock response"))
    return client


@pytest.fixture
def mock_llm_classes(mock_llm_client):
    """Patch all LLM client classes to return mock clients."""
    with patch("llm.client_factory.ChatOllama", return_value=mock_llm_client) as ollama, \
         patch("llm.client_factory.ChatGoogleGenerativeAI", return_value=mock_llm_client) as google, \
         patch("llm.client_factory.ChatMistralAI", return_value=mock_llm_client) as mistral, \
         patch("llm.client_factory.ChatOpenAI", return_value=mock_llm_client) as openai:
        yield {
            "Ollama": ollama,
            "Google": google,
            "Mistral": mistral,
            "OpenRouter": openai,
            "client": mock_llm_client,
        }
