from unittest.mock import patch

import pytest

from src.auth import RedditOAuth


class TestRedditOAuth:
    """Test Reddit OAuth handler methods."""

    def test_generate_auth_url(self):
        """Test OAuth URL generation returns valid components."""
        handler = RedditOAuth()

        auth_url, state, code_verifier = handler.generate_auth_url()

        # Basic checks
        assert "reddit.com" in auth_url
        assert "client_id=" in auth_url
        assert "redirect_uri=" in auth_url
        assert "state=" in auth_url
        assert len(state) > 10  # State should be reasonably long
        assert len(code_verifier) > 10  # Code verifier should exist

    @pytest.mark.asyncio
    async def test_exchange_code_for_tokens(self):
        """Test code exchange returns token dictionary."""
        handler = RedditOAuth()

        # Mock the HTTP request
        mock_response = {
            "access_token": "mock_access_token_123",
            "refresh_token": "mock_refresh_token_456",
            "token_type": "bearer",
        }

        with patch.object(handler, "exchange_code_for_tokens", return_value=mock_response):
            result = await handler.exchange_code_for_tokens("mock_code", "mock_verifier")

            assert "access_token" in result
            assert "refresh_token" in result
            assert result["access_token"] == "mock_access_token_123"

    @pytest.mark.asyncio
    async def test_get_user_info(self):
        """Test user info retrieval returns user data."""
        handler = RedditOAuth()

        # Mock user data
        mock_user_info = {"name": "test_reddit_user", "id": "abc123", "created_utc": 1234567890}

        with patch.object(handler, "get_user_info", return_value=mock_user_info):
            result = await handler.get_user_info("mock_access_token")

            assert "name" in result
            assert "id" in result
            assert result["name"] == "test_reddit_user"
