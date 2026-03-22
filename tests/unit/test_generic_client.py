"""
Tests for prospectio_chatbot/llm/generic_client.py - LLMGenericClient base class.
"""
from llm.generic_client import LLMGenericClient
from langchain_core.language_models.chat_models import BaseChatModel


class TestLLMGenericClient:
    """Tests for LLMGenericClient class definition and inheritance."""

    def test_inherits_base_chat_model(self):
        assert issubclass(LLMGenericClient, BaseChatModel)

    def test_class_defines_init(self):
        assert hasattr(LLMGenericClient, "__init__")
        assert callable(LLMGenericClient.__init__)
