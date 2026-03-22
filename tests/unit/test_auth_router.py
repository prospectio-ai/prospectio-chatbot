"""
Tests for prospectio_chatbot/api/auth/auth.py - Auth router token endpoint.
"""
import json
import uuid
import pytest
from unittest.mock import patch
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse


class TestGetToken:
    """Test suite for the get_token endpoint."""

    @pytest.mark.asyncio
    async def test_get_token_returns_token_and_session_id(self, mock_env_config):
        """Should return a JSON response with token and session_id."""
        from api.auth.auth import get_token

        response = await get_token()

        assert isinstance(response, JSONResponse)
        assert response.status_code == 200
        data = json.loads(response.body.decode())
        assert "token" in data
        assert "session_id" in data
        assert isinstance(data["token"], str)
        assert len(data["token"]) > 0
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0

    @pytest.mark.asyncio
    async def test_get_token_generates_unique_sessions(self, mock_env_config):
        """Each call should produce a unique session_id."""
        from api.auth.auth import get_token

        response1 = await get_token()
        response2 = await get_token()

        data1 = json.loads(response1.body.decode())
        data2 = json.loads(response2.body.decode())

        assert data1["session_id"] != data2["session_id"]
        assert data1["token"] != data2["token"]

    @pytest.mark.asyncio
    async def test_get_token_returns_valid_uuid_session_id(self, mock_env_config):
        """Session ID should be a valid UUID4 string."""
        from api.auth.auth import get_token

        response = await get_token()
        data = json.loads(response.body.decode())

        # Should not raise ValueError if it's a valid UUID
        parsed_uuid = uuid.UUID(data["session_id"], version=4)
        assert str(parsed_uuid) == data["session_id"]

    @pytest.mark.asyncio
    async def test_get_token_raises_500_on_jwt_error(self, mock_env_config):
        """Should raise HTTPException 500 if create_jwt fails."""
        from api.auth.auth import get_token

        with patch("api.auth.auth.create_jwt", side_effect=RuntimeError("JWT failure")):
            with pytest.raises(HTTPException) as exc_info:
                await get_token()
            assert exc_info.value.status_code == 500
            assert "Error occurred while generating token" in exc_info.value.detail


class TestAuthRouter:
    """Test suite for the auth router configuration."""

    def test_auth_router_is_api_router(self, mock_env_config):
        """auth_router should be a FastAPI APIRouter instance."""
        from api.auth.auth import auth_router

        assert isinstance(auth_router, APIRouter)
