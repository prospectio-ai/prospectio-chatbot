"""
Tests for prospectio_chatbot/prompts/prompt_loader.py - Prompt file loading and fallback.
"""
from prompts.prompt_loader import PromptLoader


class TestPromptLoader:
    def test_load_prompt_prospectio(self):
        loader = PromptLoader()
        prompt = loader.load_prompt("Prospectio")
        # The file exists at prompts/prospectio/system.md - it should load successfully
        assert len(prompt) > 0
        assert prompt != "You are a helpful AI assistant."

    def test_load_prompt_unknown_profile_returns_fallback(self):
        loader = PromptLoader()
        prompt = loader.load_prompt("UnknownProfile")
        assert prompt == "You are a helpful AI assistant."

    def test_prompt_mapping_has_prospectio(self):
        assert "Prospectio" in PromptLoader.prompt_mapping
        assert PromptLoader.prompt_mapping["Prospectio"] == "prospectio/system"
