"""
Authentication middleware for FeedShift request handling.
"""

import threading
import time
from typing import Any, Dict, Optional

from .tokens import TokenManager


class AuthMiddleware:
    """Authentication middleware for managing user sessions."""

    def __init__(self):
        """Initialize auth middleware with token manager."""
        self.token_manager = TokenManager()
        self.current_session: Optional[Dict[str, Any]] = None
        self._oauth_sessions: Dict[str, Dict] = {}

    def set_session(self, access_token: str, refresh_token: str, user_info: Dict[str, Any]) -> bool:
        """Set current user session from OAuth tokens and user info."""
        access_payload = self.token_manager.verify_token(access_token)

        if access_payload:
            self.current_session = {
                "user_id": access_payload.get("user_id"),
                "reddit_username": access_payload.get("reddit_username"),
                "reddit_id": access_payload.get("reddit_id"),
                "access_token": access_token,
                "refresh_token": refresh_token,
                "authenticated": True,
                "user_info": user_info,
                "session_created": access_payload.get("iat"),
            }
            return True

        return False

    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user information."""
        if not self.current_session:
            return None

        # Check if access token is still valid
        if self.token_manager.is_token_expired(self.current_session.get("access_token", "")):
            # Try to refresh the token
            if self.refresh_current_session():
                return self.current_session
            else:
                # Refresh failed, clear session
                self.clear_session()
                return None

        return self.current_session

    def is_authenticated(self) -> bool:
        """Check if current user is authenticated with valid session."""
        user = self.get_current_user()
        return user is not None and user.get("authenticated", False)

    def refresh_current_session(self) -> bool:
        """Refresh current session using refresh token."""
        if not self.current_session:
            return False

        refresh_token = self.current_session.get("refresh_token")
        if not refresh_token:
            return False

        new_access_token = self.token_manager.refresh_access_token(refresh_token)
        if new_access_token:
            self.current_session["access_token"] = new_access_token
            return True

        return False

    def clear_session(self):
        """Clear current user session (logout)."""
        self.current_session = None

    def get_user_context(self) -> Dict[str, Any]:
        """Get user context for recommendation engine."""
        user = self.get_current_user()
        if not user:
            return {"anonymous": True}

        return {
            "anonymous": False,
            "user_id": user.get("user_id"),
            "reddit_username": user.get("reddit_username"),
            "authenticated": True,
        }

    def store_oauth_session(self, session_data: Dict) -> None:
        """Store OAuth state and verifier for current session."""
        # Use a session identifier (you might have this in your existing middleware)
        session_id = self.get_current_session_id()  # Implement this based on your session handling

        # Add expiration (10 minutes)
        session_data["expires_at"] = time.time() + 600

        self._oauth_sessions[session_id] = session_data
        print(f"📝 Stored OAuth session for {session_id}")

    def get_oauth_session(self) -> Optional[Dict]:
        """Retrieve OAuth state and verifier for current session."""
        session_id = self.get_current_session_id()

        if session_id not in self._oauth_sessions:
            print(f"❌ No OAuth session found for {session_id}")
            return None

        session_data = self._oauth_sessions[session_id]

        # Check expiration
        if time.time() > session_data.get("expires_at", 0):
            print(f"⏰ OAuth session expired for {session_id}")
            del self._oauth_sessions[session_id]
            return None

        print(f"✅ Retrieved OAuth session for {session_id}")
        return session_data

    def clear_oauth_session(self) -> None:
        """Clear OAuth session after use or on error."""
        session_id = self.get_current_session_id()

        if session_id in self._oauth_sessions:
            del self._oauth_sessions[session_id]
            print(f"🧹 Cleared OAuth session for {session_id}")

    @staticmethod
    def get_current_session_id() -> str:
        """Get current session identifier - implement based on your session system."""
        return str(threading.current_thread().ident)


# Global auth middleware instance
auth_middleware = AuthMiddleware()
