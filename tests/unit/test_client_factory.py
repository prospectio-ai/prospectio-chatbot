"""
Tests for prospectio_chatbot/llm/client_factory.py - LLM client creation with mocked backends.
"""
import pytest
from llm.client_factory import LLMClientFactory


class TestLLMClientFactory:
    def test_parses_ollama_model(self, mock_env_config, mock_llm_classes):
        factory = LLMClientFactory(model="Ollama/llama3", temperature=0.5)
        client = factory.create_client()

        mock_llm_classes["Ollama"].assert_called_once_with(
            model="llama3",
            temperature=0.5,
            base_url="http://localhost:11434",
        )
        assert client is mock_llm_classes["client"]

    def test_parses_google_model(self, mock_env_config, mock_llm_classes):
        factory = LLMClientFactory(model="Google/gemini-pro", temperature=0.2)
        client = factory.create_client()

        mock_llm_classes["Google"].assert_called_once_with(
            model="gemini-pro",
            temperature=0.2,
        )
        assert client is mock_llm_classes["client"]

    def test_parses_mistral_model(self, mock_env_config, mock_llm_classes):
        factory = LLMClientFactory(model="Mistral/mistral-large", temperature=0.0)
        client = factory.create_client()

        mock_llm_classes["Mistral"].assert_called_once_with(
            model="mistral-large",
            temperature=0.0,
        )
        assert client is mock_llm_classes["client"]

    def test_parses_openrouter_model(self, mock_env_config, mock_llm_classes):
        factory = LLMClientFactory(model="OpenRouter/anthropic/claude-3", temperature=0.7)
        client = factory.create_client()

        mock_llm_classes["OpenRouter"].assert_called_once_with(
            model="anthropic/claude-3",
            temperature=0.7,
            api_key="fake-openrouter-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert client is mock_llm_classes["client"]

    def test_invalid_category_raises(self, mock_env_config, mock_llm_classes):
        factory = LLMClientFactory(model="Unknown/some-model", temperature=0.0)
        with pytest.raises(TypeError):
            factory.create_client()

    def test_model_mapping_keys(self, mock_env_config):
        factory = LLMClientFactory(model="Ollama/test", temperature=0.0)
        assert set(factory.model_mapping.keys()) == {"Ollama", "Google", "Mistral", "OpenRouter"}

    def test_preserves_model_with_slashes(self, mock_env_config, mock_llm_classes):
        """OpenRouter models often have org/model format after the category prefix."""
        factory = LLMClientFactory(model="OpenRouter/meta/llama-3-70b", temperature=0.0)
        factory.create_client()

        mock_llm_classes["OpenRouter"].assert_called_once_with(
            model="meta/llama-3-70b",
            temperature=0.0,
            api_key="fake-openrouter-key",
            base_url="https://openrouter.ai/api/v1",
        )
