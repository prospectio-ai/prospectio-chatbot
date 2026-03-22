"""
Tests for prospectio_chatbot/graphs/graph_params.py - GraphParams data holder.
"""
from graphs.graph_params import GraphParams


class TestGraphParams:
    def test_default_values(self):
        params = GraphParams()
        assert params.agent == ""
        assert params.model == ""
        assert params.temperature == 0.0
        assert params.embeddings == ""
        assert params.tools_list == []

    def test_custom_values(self):
        params = GraphParams(
            agent="Prospectio",
            model="Ollama/llama3",
            temperature=0.5,
            embeddings="nomic-embed-text",
            tools_list=["tool1", "tool2"],
        )
        assert params.agent == "Prospectio"
        assert params.model == "Ollama/llama3"
        assert params.temperature == 0.5
        assert params.embeddings == "nomic-embed-text"
        assert params.tools_list == ["tool1", "tool2"]

    def test_attributes_are_mutable(self):
        params = GraphParams()
        params.agent = "Prospectio"
        params.model = "Google/gemini-pro"
        params.temperature = 0.8
        assert params.agent == "Prospectio"
        assert params.model == "Google/gemini-pro"
        assert params.temperature == 0.8
