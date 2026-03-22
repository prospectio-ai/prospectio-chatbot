"""
Tests for prospectio_chatbot/profiles/chat_profiles.py - Chat profiles configuration.
"""
import sys
import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_chainlit(monkeypatch):
    """Mock chainlit module before importing chat_profiles."""
    mock_cl = MagicMock()

    class FakeChatProfile:
        def __init__(self, name, markdown_description, icon):
            self.name = name
            self.markdown_description = markdown_description
            self.icon = icon

    mock_cl.ChatProfile = FakeChatProfile
    monkeypatch.setitem(sys.modules, "chainlit", mock_cl)


class TestChatProfiles:
    """Test suite for ChatProfiles class."""

    def _load_module(self):
        """Reload the chat_profiles module with mocked chainlit."""
        import importlib
        import profiles.chat_profiles as mod
        importlib.reload(mod)
        return mod

    def test_get_chat_profiles_returns_list(self):
        """get_chat_profiles should return a non-empty list."""
        mod = self._load_module()
        cp = mod.ChatProfiles()
        result = cp.get_chat_profiles()

        assert isinstance(result, list)
        assert len(result) > 0

    def test_profiles_contain_prospectio(self):
        """The profiles list should contain a profile named 'Prospectio'."""
        mod = self._load_module()
        profile_names = [p.name for p in mod.profiles]
        assert "Prospectio" in profile_names

    def test_prospectio_profile_has_required_attributes(self):
        """The Prospectio profile should have name, description, and icon."""
        mod = self._load_module()
        prospectio = [p for p in mod.profiles if p.name == "Prospectio"]
        assert len(prospectio) == 1
        profile = prospectio[0]
        assert profile.name == "Prospectio"
        assert profile.markdown_description is not None
        assert profile.icon is not None

    def test_get_chat_profiles_returns_same_as_module_level(self):
        """ChatProfiles.get_chat_profiles() should return the module-level profiles list."""
        mod = self._load_module()
        cp = mod.ChatProfiles()
        assert cp.get_chat_profiles() is mod.profiles
