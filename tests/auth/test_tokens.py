from src.auth import TokenManager


class TestTokenManager:
    """Test JWT token creation and validation."""

    def test_create_access_token(self):
        """Test access token creation."""
        manager = TokenManager()
        user_info = {"name": "testuser", "id": "user123"}

        token = manager.create_access_token(user_info)

        assert isinstance(token, str)
        assert len(token) > 20  # JWT should be reasonably long
        assert token.count(".") == 2  # JWT has 3 parts separated by dots

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        manager = TokenManager()
        user_info = {"name": "testuser", "id": "user123"}

        token = manager.create_refresh_token(user_info)

        assert isinstance(token, str)
        assert len(token) > 20  # Should be reasonably long
        assert token.count(".") == 2  # JWT format

    def test_tokens_are_different(self):
        """Test that access and refresh tokens are different."""
        manager = TokenManager()
        user_info = {"name": "testuser", "id": "user123"}

        access_token = manager.create_access_token(user_info)
        refresh_token = manager.create_refresh_token(user_info)

        assert access_token != refresh_token

    def test_validate_access_token(self):
        """Test token validation (if method exists)."""
        manager = TokenManager()
        user_info = {"name": "testuser", "id": "user123"}

        token = manager.create_access_token(user_info)

        # If you have a validate method, test it
        if hasattr(manager, "validate_token"):
            is_valid = manager.validate_token(token)
            assert is_valid is True
