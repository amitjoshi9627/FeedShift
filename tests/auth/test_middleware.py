from src.auth.middleware import auth_middleware


class TestAuthMiddleware:
    """Test authentication middleware session management."""

    @staticmethod
    def setup_method():
        """Clear session before each test."""
        auth_middleware.clear_session()

    def test_set_and_get_oauth_session(self):
        """Test setting and retrieving session data."""
        # Set session
        access_token = "test_access_token"
        refresh_token = "test_refresh_token"
        user_info = {"name": "testuser", "id": "123"}

        auth_middleware.set_session(access_token, refresh_token, user_info)

        # Get session
        session = auth_middleware.get_oauth_session()

        assert session is not None
        assert session["access_token"] == access_token
        assert session["refresh_token"] == refresh_token
        assert session["user_info"]["name"] == "testuser"

    def test_clear_session(self):
        """Test session clearing."""
        # Set session first
        auth_middleware.set_session("token", "refresh", {"name": "test"})

        # Verify session exists
        session = auth_middleware.get_oauth_session()
        assert session is not None

        # Clear session
        auth_middleware.clear_session()

        # Verify session is gone
        session = auth_middleware.get_oauth_session()
        assert session is None

    def test_get_empty_session(self):
        """Test getting session when none exists."""
        # Ensure no session
        auth_middleware.clear_session()

        session = auth_middleware.get_oauth_session()
        assert session is None

    def test_oauth_session_storage(self):
        """Test OAuth session storage (if implemented)."""
        if hasattr(auth_middleware, "store_oauth_session"):
            # Test OAuth session storage
            oauth_data = {"oauth_state": "test_state_123", "oauth_code_verifier": "test_verifier_456"}

            auth_middleware.store_oauth_session(oauth_data)

            retrieved = auth_middleware.get_oauth_session()
            assert retrieved is not None
            assert retrieved["oauth_state"] == "test_state_123"

            # Clean up
            auth_middleware.clear_oauth_session()
            retrieved = auth_middleware.get_oauth_session()
            assert retrieved is None
