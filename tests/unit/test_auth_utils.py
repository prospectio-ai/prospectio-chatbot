"""
Tests for prospectio_chatbot/api/auth/utils.py - JWT token creation.
"""
import jwt
from api.auth.utils import create_jwt


class TestCreateJwt:
    def test_creates_valid_jwt(self, mock_env_config):
        token = create_jwt("session-123", {"name": "Test User"})
        # Decode without verification to check payload structure
        payload = jwt.decode(token, "test-secret-key-for-jwt", algorithms=["HS256"])
        assert payload["identifier"] == "session-123"
        assert payload["metadata"] == {"name": "Test User"}
        assert "exp" in payload

    def test_jwt_contains_expiration(self, mock_env_config):
        token = create_jwt("session-456", {"name": "Another"})
        payload = jwt.decode(token, "test-secret-key-for-jwt", algorithms=["HS256"])
        assert payload["exp"] is not None

    def test_different_identifiers_produce_different_tokens(self, mock_env_config):
        token1 = create_jwt("session-1", {"name": "User1"})
        token2 = create_jwt("session-2", {"name": "User2"})
        assert token1 != token2
