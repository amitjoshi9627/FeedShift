import base64
import hashlib
import logging
import os
import secrets
import urllib.parse
from typing import Tuple

import httpx
from dotenv import load_dotenv

from src.config.paths import ENV_FILE_PATH

# Configure logger
logger = logging.getLogger(__name__)


class RedditOAuth:
    """Reddit OAuth handler for FeedShift user authentication."""

    def __init__(self):
        """Initialize OAuth handler with Reddit app credentials."""

        load_dotenv(dotenv_path=ENV_FILE_PATH)
        logger.debug(f"Loaded environment variables from {ENV_FILE_PATH}")

        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.redirect_uri = os.getenv("REDDIT_REDIRECT_URI", "http://localhost:3000/auth/callback")
        self.user_agent = os.getenv("REDDIT_USER_AGENT", "FeedShift:v0.8.0")

        if not self.client_id or not self.client_secret:
            raise ValueError("Reddit OAuth credentials missing from environment")

    @staticmethod
    def generate_pkce_pair() -> Tuple[str, str]:
        """Generate PKCE code verifier and challenge for enhanced security."""
        # Generate code verifier (random string)
        code_verifier = base64.urlsafe_b64encode(os.urandom(40)).decode("utf-8").rstrip("=")

        # Generate code challenge (SHA256 hash of verifier)
        code_challenge = hashlib.sha256(code_verifier.encode("utf-8")).digest()
        code_challenge = base64.urlsafe_b64encode(code_challenge).decode("utf-8").rstrip("=")

        return code_verifier, code_challenge

    def generate_auth_url(self) -> Tuple[str, str, str]:
        """Generate Reddit OAuth authorization URL with PKCE."""
        # Generate CSRF protection state
        state = secrets.token_urlsafe(32)

        # Generate PKCE parameters
        code_verifier, code_challenge = self.generate_pkce_pair()

        # Build authorization parameters
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "state": state,
            "redirect_uri": self.redirect_uri,
            "duration": "permanent",
            "scope": "identity read history",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        base_url = "https://www.reddit.com/api/v1/authorize"
        auth_url = f"{base_url}?{urllib.parse.urlencode(params)}"

        return auth_url, state, code_verifier

    async def exchange_code_for_tokens(self, code: str, code_verifier: str) -> dict[str, any]:
        """Exchange authorization code for access and refresh tokens."""
        token_url = "https://www.reddit.com/api/v1/access_token"

        headers = {"User-Agent": self.user_agent, "Content-Type": "application/x-www-form-urlencoded"}

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": code_verifier,
        }

        auth = (self.client_id, self.client_secret)

        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, headers=headers, data=data, auth=auth, timeout=30.0)

            if response.status_code != 200:
                raise Exception(f"Token exchange failed: {response.status_code} - {response.text}")

            return response.json()

    async def refresh_access_token(self, refresh_token: str) -> dict[str, any]:
        """Refresh expired access token using refresh token."""
        token_url = "https://www.reddit.com/api/v1/access_token"

        headers = {"User-Agent": self.user_agent, "Content-Type": "application/x-www-form-urlencoded"}

        data = {"grant_type": "refresh_token", "refresh_token": refresh_token}

        auth = (self.client_id, self.client_secret)

        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, headers=headers, data=data, auth=auth, timeout=30.0)

            if response.status_code != 200:
                raise Exception(f"Token refresh failed: {response.text}")

            return response.json()

    async def get_user_info(self, access_token: str) -> dict[str, any]:
        """Get authenticated user information from Reddit API."""
        headers = {"Authorization": f"Bearer {access_token}", "User-Agent": self.user_agent}

        async with httpx.AsyncClient() as client:
            response = await client.get("https://oauth.reddit.com/api/v1/me", headers=headers, timeout=30.0)

            if response.status_code != 200:
                raise Exception(f"Failed to get user info: {response.text}")

            return response.json()
