"""
Tests for prospectio_chatbot/graphs/prospectio/chains/generate.py - GenerateChain construction.
"""
from unittest.mock import patch, MagicMock
from graphs.prospectio.chains.generate import GenerateChain


class TestGenerateChain:
    """Tests for GenerateChain initialization and chain composition."""

    def test_init_creates_chain(self, mock_env_config, mock_llm_classes):
        tools = [MagicMock()]
        chain = GenerateChain(
            model="Ollama/llama3",
            temperature=0.5,
            prompt="You are a helpful assistant.",
            tools_list=tools,
        )
        assert chain.chain is not None

    def test_init_calls_llm_factory(self, mock_env_config, mock_llm_classes):
        tools = [MagicMock()]
        with patch("graphs.prospectio.chains.generate.LLMClientFactory") as mock_factory:
            mock_client = MagicMock()
            mock_client.bind_tools.return_value = mock_client
            mock_factory.return_value.create_client.return_value = mock_client

            chain = GenerateChain(
                model="Google/gemini-pro",
                temperature=0.3,
                prompt="System prompt here.",
                tools_list=tools,
            )

            mock_factory.assert_called_once_with(
                model="Google/gemini-pro", temperature=0.3
            )
            mock_factory.return_value.create_client.assert_called_once()

    def test_init_binds_tools_to_llm(self, mock_env_config, mock_llm_classes):
        tools = [MagicMock(), MagicMock()]
        with patch("graphs.prospectio.chains.generate.LLMClientFactory") as mock_factory:
            mock_client = MagicMock()
            mock_client.bind_tools.return_value = mock_client
            mock_factory.return_value.create_client.return_value = mock_client

            GenerateChain(
                model="Ollama/llama3",
                temperature=0.0,
                prompt="Test prompt",
                tools_list=tools,
            )

            mock_client.bind_tools.assert_called_once_with(tools)
