"""
Tests for prospectio_chatbot/graphs/prospectio/graph.py - ProspectioGraph construction.
"""
import pytest
from unittest.mock import patch, MagicMock
from graphs.prospectio.graph import ProspectioGraph
from graphs.generic_graph import GenericGraph
from graphs.graph_params import GraphParams


class TestProspectioGraph:
    """Tests for ProspectioGraph initialization, construction, and compilation."""

    @pytest.fixture
    def graph_params(self):
        return GraphParams(
            agent="Prospectio",
            model="Ollama/llama3",
            temperature=0.5,
            tools_list=[MagicMock()],
        )

    @pytest.fixture
    def mock_dependencies(self, mock_env_config, mock_llm_classes):
        """Patch ProspectioNodes and StateGraph to avoid real LLM/tool instantiation."""
        with patch("graphs.prospectio.graph.ProspectioNodes") as mock_nodes_cls, \
             patch("graphs.prospectio.graph.StateGraph") as mock_sg_cls:
            mock_sg_instance = MagicMock()
            mock_sg_cls.return_value = mock_sg_instance
            mock_nodes_instance = MagicMock()
            mock_nodes_cls.return_value = mock_nodes_instance
            yield {
                "nodes_cls": mock_nodes_cls,
                "nodes": mock_nodes_instance,
                "sg_cls": mock_sg_cls,
                "sg": mock_sg_instance,
            }

    def test_inherits_generic_graph(self):
        assert issubclass(ProspectioGraph, GenericGraph)

    def test_init_creates_state_graph(self, graph_params, mock_dependencies):
        ProspectioGraph(graph_params)
        mock_dependencies["sg_cls"].assert_called_once()

    def test_init_creates_nodes(self, graph_params, mock_dependencies):
        ProspectioGraph(graph_params)
        mock_dependencies["nodes_cls"].assert_called_once_with(graph_params)

    def test_construct_graph_adds_nodes(self, graph_params, mock_dependencies):
        ProspectioGraph(graph_params)
        sg = mock_dependencies["sg"]
        # Two nodes should be added: call_model and tools
        assert sg.add_node.call_count == 2
        node_names = [call.args[0] for call in sg.add_node.call_args_list]
        assert "call_model" in node_names
        assert "tools" in node_names

    def test_construct_graph_adds_edges(self, graph_params, mock_dependencies):
        ProspectioGraph(graph_params)
        sg = mock_dependencies["sg"]
        # One regular edge from START to call_model, one from tools to call_model
        assert sg.add_edge.call_count == 2
        # One conditional edge from call_model
        sg.add_conditional_edges.assert_called_once()

    def test_get_graph_returns_compiled(self, graph_params, mock_dependencies):
        graph = ProspectioGraph(graph_params)
        compiled = MagicMock()
        mock_dependencies["sg"].compile.return_value = compiled

        result = graph.get_graph()
        mock_dependencies["sg"].compile.assert_called_once()
        assert result is compiled
