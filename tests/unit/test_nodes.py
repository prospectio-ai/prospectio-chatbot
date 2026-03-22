"""
Tests for prospectio_chatbot/graphs/prospectio/nodes.py - ProspectioNodes logic.
"""
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from graphs.prospectio.nodes import ProspectioNodes
from graphs.graph_params import GraphParams
from langgraph.graph import END


class TestProspectioNodes:
    """Tests for ProspectioNodes initialization and methods."""

    @pytest.fixture
    def graph_params(self):
        return GraphParams(
            agent="Prospectio",
            model="Ollama/llama3",
            temperature=0.5,
            tools_list=[MagicMock()],
        )

    @pytest.fixture
    def nodes(self, mock_env_config, mock_llm_classes, graph_params):
        with patch("graphs.prospectio.nodes.ToolNode") as mock_tool_node:
            mock_tool_node.return_value = MagicMock()
            instance = ProspectioNodes(graph_params)
        return instance

    def test_init_sets_graph_params(self, mock_env_config, mock_llm_classes):
        params = GraphParams(
            agent="Prospectio",
            model="Ollama/llama3",
            temperature=0.5,
            tools_list=[MagicMock()],
        )
        with patch("graphs.prospectio.nodes.ToolNode"):
            node = ProspectioNodes(params)
        assert node.graph_params is params

    def test_init_creates_tool_node(self, mock_env_config, mock_llm_classes):
        tools = [MagicMock(), MagicMock()]
        params = GraphParams(
            agent="Prospectio",
            model="Ollama/llama3",
            temperature=0.0,
            tools_list=tools,
        )
        with patch("graphs.prospectio.nodes.ToolNode") as mock_tool_node_cls:
            ProspectioNodes(params)
            mock_tool_node_cls.assert_called_once_with(tools)

    def test_init_creates_generate_chain(self, mock_env_config, mock_llm_classes):
        params = GraphParams(
            agent="Prospectio",
            model="Ollama/llama3",
            temperature=0.3,
            tools_list=[MagicMock()],
        )
        with patch("graphs.prospectio.nodes.ToolNode"), \
             patch("graphs.prospectio.nodes.GenerateChain") as mock_chain_cls:
            ProspectioNodes(params)
            mock_chain_cls.assert_called_once()
            call_kwargs = mock_chain_cls.call_args
            assert call_kwargs.kwargs["model"] == "Ollama/llama3"
            assert call_kwargs.kwargs["temperature"] == 0.3

    def test_should_continue_returns_tools_when_tool_calls_present(self, nodes):
        last_message = MagicMock()
        last_message.tool_calls = [{"name": "search", "args": {}}]
        state = {"messages": [last_message]}

        result = nodes.should_continue(state)
        assert result == "tools"

    def test_should_continue_returns_end_when_no_tool_calls(self, nodes):
        last_message = MagicMock()
        last_message.tool_calls = []
        state = {"messages": [last_message]}

        result = nodes.should_continue(state)
        assert result == END

    def test_should_continue_returns_end_when_no_tool_calls_attr(self, nodes):
        last_message = MagicMock(spec=[])  # no attributes at all
        state = {"messages": [last_message]}

        result = nodes.should_continue(state)
        assert result == END

    def test_call_tools_invokes_tool_node(self, nodes):
        expected_response = {"messages": [MagicMock()]}
        nodes.tool_node.ainvoke = AsyncMock(return_value=expected_response)
        state = {"messages": [MagicMock()]}

        result = asyncio.run(nodes.call_tools(state))
        nodes.tool_node.ainvoke.assert_called_once_with({"messages": state["messages"]})
        assert result == expected_response

    def test_call_model_invokes_chain(self, nodes):
        mock_response = MagicMock(content="Hello!")
        nodes.generate_chain = MagicMock()
        nodes.generate_chain.chain.ainvoke = AsyncMock(return_value=mock_response)
        state = {"messages": [MagicMock()]}

        result = asyncio.run(nodes.call_model(state))
        nodes.generate_chain.chain.ainvoke.assert_called_once_with(
            {"messages": state["messages"]}
        )
        assert result == {"messages": [mock_response]}
