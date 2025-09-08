"""
FeedShift Authentication Module

Provides Reddit OAuth integration, JWT session management,
and authentication middleware for personalized recommendations.
"""

from .middleware import AuthMiddleware
from .oauth import RedditOAuth
from .tokens import TokenManager

__all__ = ["RedditOAuth", "TokenManager", "AuthMiddleware"]
