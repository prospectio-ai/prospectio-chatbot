"""
Tests for prospectio_chatbot/graphs/graph_factory.py - Graph creation routing.
"""
import pytest
from unittest.mock import patch, MagicMock
from graphs.graph_factory import GraphFactory
from graphs.graph_params import GraphParams


class TestGraphFactory:
    def test_create_graph_prospectio(self):
        params = GraphParams(agent="Prospectio", model="Ollama/llama3", temperature=0.0)
        mock_graph_instance = MagicMock()

        with patch("graphs.graph_factory.ProspectioGraph", return_value=mock_graph_instance) as mock_cls:
            factory = GraphFactory(params)
            result = factory.create_graph()

        mock_cls.assert_called_once_with(params)
        assert result is mock_graph_instance

    def test_create_graph_invalid_agent_raises(self):
        params = GraphParams(agent="NonExistent", model="Ollama/llama3", temperature=0.0)
        factory = GraphFactory(params)

        with pytest.raises(ValueError, match="Invalid graph name: NonExistent"):
            factory.create_graph()

    def test_graph_mapping_contains_prospectio(self):
        params = GraphParams()
        factory = GraphFactory(params)
        assert "Prospectio" in factory.graph_mapping
