"""
Tests for prospectio_chatbot/settings/chat_settings.py - Chat settings configuration.
"""
import pytest
from unittest.mock import patch


class TestChatSettings:
    """Test suite for ChatSettings class."""

    def test_get_chat_settings_returns_two_widgets(self, mock_env_config):
        """get_chat_settings should return a list with Model selector and Temperature slider."""
        from settings.chat_settings import ChatSettings
        cs = ChatSettings()
        settings = cs.get_chat_settings()

        assert isinstance(settings, list)
        assert len(settings) == 2

    def test_first_widget_is_model_selector(self, mock_env_config):
        """First widget should be a Select for LLM Model."""
        from settings.chat_settings import ChatSettings
        from chainlit.input_widget import Select

        cs = ChatSettings()
        settings = cs.get_chat_settings()
        model_widget = settings[0]

        assert isinstance(model_widget, Select)
        assert model_widget.id == "Model"
        assert model_widget.label == "LLM Model"

    def test_second_widget_is_temperature_slider(self, mock_env_config):
        """Second widget should be a Slider for Temperature."""
        from settings.chat_settings import ChatSettings
        from chainlit.input_widget import Slider

        cs = ChatSettings()
        settings = cs.get_chat_settings()
        temp_widget = settings[1]

        assert isinstance(temp_widget, Slider)
        assert temp_widget.id == "Temperature"
        assert temp_widget.label == "Model Temperature"

    def test_temperature_slider_range(self, mock_env_config):
        """Temperature slider should range from 0.0 to 1.0 with step 0.1."""
        from settings.chat_settings import ChatSettings
        from chainlit.input_widget import Slider

        cs = ChatSettings()
        settings = cs.get_chat_settings()
        temp_widget = settings[1]

        assert temp_widget.initial == 0.0
        assert temp_widget.min == 0.0
        assert temp_widget.max == 1.0
        assert temp_widget.step == 0.1

    def test_model_selector_uses_models_from_config(self, mock_env_config):
        """Model selector values should come from ChainlitSettings.MODELS_LIST."""
        from settings.chat_settings import ChatSettings

        cs = ChatSettings()
        settings = cs.get_chat_settings()
        model_widget = settings[0]

        assert model_widget.values == ["Ollama/llama3", "Google/gemini-pro"]

    def test_models_list_loaded_from_config(self, mock_env_config):
        """ChatSettings.models_list should be loaded from ChainlitSettings."""
        from settings.chat_settings import ChatSettings

        cs = ChatSettings()
        assert isinstance(cs.models_list, list)
        assert len(cs.models_list) > 0
