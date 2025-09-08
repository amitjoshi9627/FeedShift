"""
JWT Token management for FeedShift user sessions.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import jwt
from dotenv import load_dotenv

from src.config.paths import ENV_FILE_PATH

# Configure logger
logger = logging.getLogger(__name__)


class TokenManager:
    """JWT Token manager for secure user session handling."""

    def __init__(self):
        """Initialize token manager with secure configuration."""

        load_dotenv(dotenv_path=ENV_FILE_PATH)
        logger.debug(f"Loaded environment variables from {ENV_FILE_PATH}")

        self.secret_key = os.getenv("JWT_SECRET_KEY")
        if not self.secret_key:
            raise ValueError("JWT_SECRET_KEY not found in environment variables")

        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60  # 1 hour
        self.refresh_token_expire_days = 30  # 30 days

    def create_access_token(self, user_data: Dict[str, any]) -> str:
        """Create short-lived JWT access token for API authentication."""
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "user_id": user_data.get("id"),
            "reddit_username": user_data.get("name"),
            "reddit_id": user_data.get("id"),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "iss": "FeedShift",
            "sub": user_data.get("name"),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, user_data: Dict[str, any]) -> str:
        """Create long-lived JWT refresh token for session persistence."""
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "user_id": user_data.get("id"),
            "reddit_username": user_data.get("name"),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "iss": "FeedShift",
            "sub": user_data.get("name"),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, any]]:
        """Verify and decode JWT token with comprehensive validation."""
        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": True, "verify_iss": True}
            )

            # Additional validation
            if payload.get("iss") != "FeedShift":
                return None

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            logger.error(e)
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate new access token from valid refresh token."""
        payload = self.verify_token(refresh_token)

        if not payload or payload.get("type") != "refresh":
            return None

        # Create new access token with same user data
        user_data = {"id": payload.get("user_id"), "name": payload.get("reddit_username")}

        return self.create_access_token(user_data)

    @staticmethod
    def is_token_expired(token: str) -> bool:
        """Check if token is expired without full verification."""
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            exp = payload.get("exp")
            if not exp:
                return True
            return datetime.utcnow().timestamp() > exp
        except Exception as e:
            logger.warning(f"Error: {e}")
            return True
