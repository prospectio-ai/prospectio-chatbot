"""
Tests for prospectio_chatbot/core/essentials.py - CoreEssentials class.
"""
import sys
import importlib
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture(autouse=True)
def mock_chainlit_module(monkeypatch):
    """Mock chainlit and its submodules before importing essentials."""
    mock_cl = MagicMock()
    mock_server = MagicMock()
    mock_types = MagicMock()
    monkeypatch.setitem(sys.modules, "chainlit", mock_cl)
    monkeypatch.setitem(sys.modules, "chainlit.server", mock_server)
    monkeypatch.setitem(sys.modules, "chainlit.types", mock_types)
    mock_cl.server = mock_server
    mock_cl.types = mock_types
    return mock_cl


def _reload_essentials():
    """Reload core.essentials module to pick up mocked dependencies."""
    import core.essentials as mod
    importlib.reload(mod)
    return mod


class TestCoreEssentialsInit:
    """Test suite for CoreEssentials initialization."""

    def test_nodes_mapping_contains_prospectio(self, mock_env_config):
        """nodes_mapping should contain the Prospectio agent key."""
        with patch("graphs.graph_factory.GraphFactory.__init__", return_value=None), \
             patch("graphs.graph_params.GraphParams.__init__", return_value=None):
            mod = _reload_essentials()
            assert "Prospectio" in mod.CoreEssentials.nodes_mapping
            assert mod.CoreEssentials.nodes_mapping["Prospectio"] == "call_model"

    def test_init_creates_graph_params_and_factory(self, mock_env_config):
        """__init__ should create GraphParams and GraphFactory instances."""
        mod = _reload_essentials()
        with patch.object(mod, "GraphParams") as mock_params, \
             patch.object(mod, "GraphFactory") as mock_factory:
            ce = mod.CoreEssentials()
            mock_params.assert_called_once()
            mock_factory.assert_called_once_with(mock_params.return_value)
            assert ce.graph_params is mock_params.return_value
            assert ce.graph_factory is mock_factory.return_value

    def test_init_loads_mcp_servers_from_settings(self, mock_env_config):
        """__init__ should load MCP_SERVERS from MCPSettings."""
        mod = _reload_essentials()
        with patch.object(mod, "GraphParams"), \
             patch.object(mod, "GraphFactory"):
            ce = mod.CoreEssentials()
            assert ce.mcp_servers == []


class TestSetupChat:
    """Test suite for CoreEssentials.setup_chat."""

    def test_setup_chat_sets_agent_model_temperature(self, mock_env_config, mock_chainlit_module):
        """setup_chat should configure agent, model, temperature on graph_params and session."""
        mod = _reload_essentials()
        mock_cl = mod.cl
        mock_cl.user_session.get.return_value = "Prospectio"

        with patch.object(mod, "GraphParams") as mock_params_cls, \
             patch.object(mod, "GraphFactory") as mock_factory:
            mock_params = MagicMock()
            mock_params_cls.return_value = mock_params
            mock_factory.return_value.create_graph.return_value = MagicMock()

            ce = mod.CoreEssentials()
            ce.setup_chat("Ollama/llama3", 0.5)

            assert ce.graph_params.agent == "Prospectio"
            assert ce.graph_params.model == "Ollama/llama3"
            assert ce.graph_params.temperature == 0.5
            mock_cl.user_session.set.assert_any_call("model", "Ollama/llama3")
            mock_cl.user_session.set.assert_any_call("temperature", 0.5)

    def test_setup_chat_defaults_to_prospectio_when_no_profile(self, mock_env_config, mock_chainlit_module):
        """setup_chat should use 'Prospectio' if chat_profile is not set."""
        mod = _reload_essentials()
        mock_cl = mod.cl
        mock_cl.user_session.get.return_value = None

        with patch.object(mod, "GraphParams") as mock_params_cls, \
             patch.object(mod, "GraphFactory") as mock_factory:
            mock_params = MagicMock()
            mock_params_cls.return_value = mock_params
            mock_factory.return_value.create_graph.return_value = MagicMock()

            ce = mod.CoreEssentials()
            ce.setup_chat("Google/gemini-pro", 0.0)

            assert ce.graph_params.agent == "Prospectio"


class TestCallAgent:
    """Test suite for CoreEssentials.call_agent."""

    def test_call_agent_creates_graph_and_streams(self, mock_env_config, mock_chainlit_module):
        """call_agent should create a graph and call astream with chat history."""
        mod = _reload_essentials()
        mock_cl = mod.cl
        mock_cl.user_session.get.return_value = []
        mock_cl.chat_context.to_openai.return_value = [{"role": "user", "content": "hello"}]
        mock_cl.context.session.id = "session-123"
        mock_cl.LangchainCallbackHandler.return_value = MagicMock()

        mock_graph_instance = MagicMock()
        mock_compiled = MagicMock()
        mock_graph_instance.get_graph.return_value = mock_compiled
        mock_compiled.astream.return_value = iter([])

        with patch.object(mod, "GraphParams"), \
             patch.object(mod, "GraphFactory") as mock_factory:
            mock_factory.return_value.create_graph.return_value = mock_graph_instance

            ce = mod.CoreEssentials()
            response = ce.call_agent()

            mock_compiled.astream.assert_called_once()
            call_args = mock_compiled.astream.call_args
            assert call_args[0][0] == {"messages": [{"role": "user", "content": "hello"}]}
            assert call_args[1]["stream_mode"] == ["messages", "updates"]


class TestProcessSources:
    """Test suite for CoreEssentials.process_sources."""

    def test_process_sources_extracts_sources_from_updates(self, mock_env_config, mock_chainlit_module):
        """process_sources should extract sources from 'updates' chunks."""
        mod = _reload_essentials()
        mock_cl = mod.cl
        mock_cl.Text.return_value = MagicMock()

        mock_answer = MagicMock()
        mock_answer.elements = []

        with patch.object(mod, "GraphParams"), \
             patch.object(mod, "GraphFactory"):
            ce = mod.CoreEssentials()

            chunk = ("updates", {"retrieve_sources": {"sources": ["source1.pdf", "source2.pdf"]}})
            result = ce.process_sources("retrieve_sources", chunk, mock_answer)

            assert "source1.pdf" in result
            assert "source2.pdf" in result
            assert len(mock_answer.elements) == 1

    def test_process_sources_ignores_non_updates_chunks(self, mock_env_config, mock_chainlit_module):
        """process_sources should return empty list for non-updates chunks."""
        mod = _reload_essentials()
        mock_answer = MagicMock()
        mock_answer.elements = []

        with patch.object(mod, "GraphParams"), \
             patch.object(mod, "GraphFactory"):
            ce = mod.CoreEssentials()

            chunk = ("messages", {"some_node": {"content": "text"}})
            result = ce.process_sources("retrieve_sources", chunk, mock_answer)

            assert result == []
            assert len(mock_answer.elements) == 0

    def test_process_sources_ignores_updates_without_matching_node(self, mock_env_config, mock_chainlit_module):
        """process_sources should ignore updates not matching the node_name."""
        mod = _reload_essentials()
        mock_answer = MagicMock()
        mock_answer.elements = []

        with patch.object(mod, "GraphParams"), \
             patch.object(mod, "GraphFactory"):
            ce = mod.CoreEssentials()

            chunk = ("updates", {"other_node": {"sources": ["source1.pdf"]}})
            result = ce.process_sources("retrieve_sources", chunk, mock_answer)

            assert result == []
            assert len(mock_answer.elements) == 0


class TestConnectMcpForSession:
    """Test suite for CoreEssentials.connect_mcp_for_session."""

    @pytest.mark.asyncio
    async def test_connect_mcp_returns_success_when_no_servers(self, mock_env_config, mock_chainlit_module):
        """Should return success message when MCP_SERVERS is empty."""
        mod = _reload_essentials()

        with patch.object(mod, "GraphParams"), \
             patch.object(mod, "GraphFactory"), \
             patch.object(mod, "connect_mcp", new_callable=AsyncMock):
            ce = mod.CoreEssentials()
            ce.mcp_servers = []

            result = await ce.connect_mcp_for_session()
            assert result["message"] == "Connected to MCP servers successfully."

    @pytest.mark.asyncio
    async def test_connect_mcp_calls_connect_for_each_server(self, mock_env_config, mock_chainlit_module):
        """Should call connect_mcp for each server in the list."""
        mod = _reload_essentials()
        mock_cl = mod.cl
        mock_cl.context.session.id = "session-123"
        mock_cl.context.session.user = MagicMock()

        with patch.object(mod, "GraphParams"), \
             patch.object(mod, "GraphFactory"), \
             patch.object(mod, "connect_mcp", new_callable=AsyncMock) as mock_connect, \
             patch.object(mod, "ConnectSseMCPRequest"):
            ce = mod.CoreEssentials()
            ce.mcp_servers = [
                {"clientType": "sse", "name": "server1", "url": "http://localhost:8001"},
                {"clientType": "sse", "name": "server2", "url": "http://localhost:8002"},
            ]

            result = await ce.connect_mcp_for_session()

            assert mock_connect.call_count == 2
            assert result["message"] == "Connected to MCP servers successfully."

    @pytest.mark.asyncio
    async def test_connect_mcp_returns_failure_on_error(self, mock_env_config, mock_chainlit_module):
        """Should return failure message with error details on exception."""
        mod = _reload_essentials()
        mock_cl = mod.cl
        mock_cl.context.session.id = "session-123"
        mock_cl.context.session.user = MagicMock()

        with patch.object(mod, "GraphParams"), \
             patch.object(mod, "GraphFactory"), \
             patch.object(mod, "connect_mcp", new_callable=AsyncMock, side_effect=RuntimeError("Connection failed")), \
             patch.object(mod, "ConnectSseMCPRequest"):
            ce = mod.CoreEssentials()
            ce.mcp_servers = [
                {"clientType": "sse", "name": "server1", "url": "http://localhost:8001"},
            ]

            result = await ce.connect_mcp_for_session()

            assert result["message"] == "Failed to connect to MCP servers."
            assert "Connection failed" in result["error"]
