"""
FeedShift Dashboard State Management

Handles all application state, event handlers, and business logic
for user authentication, recommendations, and UI state management.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import reflex as rx

from dashboard.constants import DefaultValues, InterestCategories, get_similarity_type_from_value
from dashboard.utils import get_recommended_posts
from src.auth import RedditOAuth, TokenManager
from src.auth.middleware import auth_middleware
from src.data.constants import SubRedditList

logger = logging.getLogger(__name__)


class FeedShiftState(rx.State):
    """
    Main state management class for the FeedShift dashboard.

    Handles all application state including user selections,
    processing status, recommendation results, and authentication.
    """

    # === Core Data State ===
    data: Optional[pd.DataFrame] = None
    saved_data_path: Union[str, Path] = ""

    # === User Selections ===
    selected_subreddit: str = ""
    toxicity_strictness: float = DefaultValues.TOXICITY_STRICTNESS
    diversity_strength: float = DefaultValues.DIVERSITY_STRENGTH
    selected_interests: List[str] = []

    # === Results State ===
    ranked_data: Optional[pd.DataFrame] = None
    ranked_posts: List[Dict[str, Any]] = []

    # === UI State ===
    subreddit_selected: bool = False
    is_processing: bool = False
    has_generated_once: bool = False

    # === Task Management ===
    _current_task_id: int = 0
    _last_processed_value: float = DefaultValues.TOXICITY_STRICTNESS
    _last_processed_diversity: float = DefaultValues.DIVERSITY_STRENGTH
    _cancelled_tasks: int = 0

    # === Authentication State ===
    is_authenticated: bool = False
    is_logging_in: bool = False
    login_error: str = ""
    reddit_username: str = ""
    user_id: str = ""
    access_token: str = ""
    refresh_token: str = ""
    oauth_state: str = ""
    oauth_code_verifier: str = ""

    # === Available Options ===
    available_interests: List[str] = InterestCategories.get_all()
    available_subreddits: List[str] = SubRedditList.get_all()

    def __init__(self, **kwargs) -> None:
        """Initialize the FeedShift state with default values."""
        logger.info("Initializing FeedShiftState")
        super().__init__(**kwargs)
        self.reset_to_initial_state()
        logger.debug("FeedShiftState initialized with default values")

    def reset_to_initial_state(self) -> None:
        """Reset all state variables to their initial default values."""
        logger.info("Resetting FeedShiftState to initial values")

        # Reset core data
        self.data = None
        self.saved_data_path = ""

        # Reset user selections
        self.selected_subreddit = ""
        self.toxicity_strictness = DefaultValues.TOXICITY_STRICTNESS
        self.diversity_strength = DefaultValues.DIVERSITY_STRENGTH
        self.selected_interests = []

        # Reset results
        self.ranked_data = None
        self.ranked_posts = []

        # Reset UI state
        self.subreddit_selected = False
        self.is_processing = False
        self.has_generated_once = False

        # Reset task management
        self._current_task_id = 0
        self._last_processed_value = DefaultValues.TOXICITY_STRICTNESS
        self._last_processed_diversity = DefaultValues.DIVERSITY_STRENGTH
        self._cancelled_tasks = 0

        logger.debug("State reset completed")

    # === Reactive Variables ===

    @rx.var
    def diversity_label(self) -> str:
        """Get human-readable diversity label based on current diversity strength."""
        similarity_type = get_similarity_type_from_value(self.diversity_strength)
        logger.debug(f"Diversity strength {self.diversity_strength} mapped to {similarity_type}")
        return similarity_type

    @rx.var
    def processing_info(self) -> str:
        """Get current processing information for display in UI."""
        interests_text = f"{len(self.selected_interests)} interests" if self.selected_interests else "No interests"
        subreddit_text = f"r/{self.selected_subreddit}" if self.selected_subreddit else "No subreddit"
        diversity_label = self.diversity_label.lower()

        info = (
            f"Processing {subreddit_text} with {interests_text}, "
            f"toxicity: {self.toxicity_strictness:.2f}, "
            f"diversity: {diversity_label} ({self.diversity_strength:.2f})"
        )

        logger.debug(f"Generated processing info: {info}")
        return info

    @rx.event
    async def debug_oauth_state(self):
        """Debug method to check OAuth state persistence."""
        print(f"🐛 Debug - oauth_state: '{self.oauth_state}'")
        print(f"🐛 Debug - oauth_code_verifier: '{self.oauth_code_verifier}'")
        print(f"🐛 Debug - is_logging_in: {self.is_logging_in}")

        # Test setting and getting
        self.oauth_state = "test_state_123"
        print(f"🐛 After setting - oauth_state: '{self.oauth_state}'")

    @rx.var
    def can_generate_recommendations(self) -> bool:
        """Check if recommendations can be generated with current state."""
        can_generate = bool(self.selected_subreddit and self.selected_subreddit.strip())
        logger.debug(f"Can generate recommendations: {can_generate}")
        return can_generate

    # === Authentication Event Handlers ===

    @rx.event
    async def initiate_reddit_login(self):
        """Start Reddit OAuth login with middleware state storage."""
        print("🔄 Starting Reddit OAuth login")

        self.is_logging_in = True
        self.login_error = ""

        try:
            oauth_handler = RedditOAuth()
            auth_url, state, code_verifier = oauth_handler.generate_auth_url()

            print(f"📝 Generated OAuth URL: {auth_url[:100]}...")
            print(f"🔑 State: {state}")
            print(f"🔐 Code Verifier: {code_verifier}")

            session_data = {"oauth_state": state, "oauth_code_verifier": code_verifier, "timestamp": time.time()}

            # Use your existing auth middleware to store OAuth state
            auth_middleware.store_oauth_session(session_data)

            print("💾 Stored OAuth state in auth middleware")
            print("🚀 Redirecting to Reddit...")

            return rx.redirect(auth_url, is_external=True)

        except Exception as e:
            print(f"❌ OAuth initiation failed: {e}")
            self.login_error = f"Failed to start login: {str(e)}"
            self.is_logging_in = False

    @rx.event
    async def process_oauth_from_url(self):
        """
        Process OAuth parameters directly from the current page URL.

        This method is called when the callback page loads via on_load
        and extracts OAuth parameters from the URL without JavaScript.
        """
        logger.info("Processing OAuth parameters from URL")

        try:
            # Access query parameters from the router
            code = self.router.page.params.get("code")
            state_param = self.router.page.params.get("state")
            error = self.router.page.params.get("error")
            print("\nRouter: ", self.router.page)

            logger.info(f"OAuth params - code: {bool(code)}, state: {bool(state_param)}, error: {error}")

            # Handle error case
            if error:
                logger.error(f"OAuth error received: {error}")
                self.login_error = f"Authentication failed: {error}"
                self.is_logging_in = False
                return rx.redirect("/")

            # Handle missing parameters
            if not code or not state_param:
                logger.error("Missing OAuth parameters")
                self.login_error = "Invalid authentication response - missing parameters"
                self.is_logging_in = False
                return rx.redirect("/")

            await self.handle_oauth_callback(code, state_param)

            # Redirect to main page after successful processing
            logger.info("OAuth processing completed, redirecting to main page")
            return rx.redirect("/")

        except Exception as e:
            logger.error(f"Error processing OAuth from URL: {e}")
            self.login_error = f"Authentication failed: {str(e)}"
            self.is_logging_in = False
            return rx.redirect("/")

    def get_router_data(self):
        """Get router data for accessing URL parameters."""
        return self.router

    @rx.event
    async def handle_oauth_callback(self, code: str, state: str):
        """Handle OAuth callback with middleware state retrieval."""
        print(f"🔍 Processing OAuth callback - code: {code[:20]}..., state: {state}")

        try:
            # 🔑 Retrieve OAuth state from auth middleware
            oauth_session = auth_middleware.get_oauth_session()

            if not oauth_session:
                raise ValueError("No OAuth session found - possible timeout or session loss")

            stored_state = oauth_session.get("oauth_state")
            stored_verifier = oauth_session.get("oauth_code_verifier")

            print(f"🔍 Retrieved from middleware - stored state: '{stored_state}'")
            print(f"🔍 Received state: '{state}'")

            # Security check: verify state parameter
            if not stored_state or state != stored_state:
                print("❌ State mismatch!")
                print(f"   Expected: '{stored_state}'")
                print(f"   Received: '{state}'")
                raise ValueError("Invalid state parameter - possible CSRF attack")

            print("✅ State parameter validated from auth middleware")

            # Exchange code for tokens using stored verifier
            oauth_handler = RedditOAuth()
            print("🔄 Exchanging code for tokens...")

            token_response = await oauth_handler.exchange_code_for_tokens(
                code, stored_verifier  # Use verifier from middleware
            )

            print("✅ Tokens received, fetching user info...")

            # Get user information
            user_info = await oauth_handler.get_user_info(token_response["access_token"])
            print(f"👤 User info retrieved: {user_info.get('name', 'Unknown')}")

            # Create internal JWT tokens
            token_manager = TokenManager()
            access_token = token_manager.create_access_token(user_info)
            refresh_token = token_manager.create_refresh_token(user_info)

            # Set session
            auth_middleware.set_session(access_token, refresh_token, user_info)

            # Update UI state
            self.is_authenticated = True
            self.reddit_username = user_info.get("name", "")
            self.user_id = user_info.get("id", "")
            self.access_token = access_token
            self.refresh_token = refresh_token

            # 🧹 Clear OAuth session after successful use
            auth_middleware.clear_oauth_session()

            print(f"✅ OAuth completed successfully for {self.reddit_username}")

        except Exception as e:
            print(f"❌ OAuth callback failed: {e}")
            self.login_error = f"Login failed: {str(e)}"
            self.is_authenticated = False

            # Clear OAuth session on error
            auth_middleware.clear_oauth_session()

        finally:
            # CRITICAL: Always reset loading state
            self.is_logging_in = False
            print("🔄 Loading state reset")

    @rx.event
    async def logout(self):
        """Logout current user and clear all authentication state."""
        logger.info("User logout requested")

        # Clear authentication state
        self.is_authenticated = False
        self.is_logging_in = False
        self.login_error = ""

        # Clear user information
        self.reddit_username = ""
        self.user_id = ""

        # Clear tokens
        self.access_token = ""
        self.refresh_token = ""
        self.oauth_state = ""
        self.oauth_code_verifier = ""

        # Clear session from middleware
        auth_middleware.clear_session()

        logger.info("User logged out successfully")

    @rx.event
    def set_subreddit(self, subreddit: str) -> Optional[Any]:
        """Set the selected subreddit and trigger auto-generation if applicable."""
        logger.info(f"Setting subreddit to: {subreddit}")

        try:
            if not subreddit or not subreddit.strip():
                logger.warning("Empty subreddit name provided")
                return

            self.selected_subreddit = subreddit.strip()
            self.subreddit_selected = True
            self.ranked_data = None
            self.ranked_posts = []

            logger.info(f"Subreddit set to r/{self.selected_subreddit}")

            # Auto-generate if we've generated recommendations before
            if self.has_generated_once:
                logger.debug("Auto-generating recommendations for new subreddit")
                self._current_task_id += 1
                current_task = self._current_task_id
                return FeedShiftState.auto_generate_recommendations(current_task)

        except Exception as e:
            logger.error(f"Error setting subreddit to {subreddit}: {e}")
            return

    @rx.event
    def toggle_interest(self, interest: str) -> Optional[Any]:
        """Toggle interest selection on/off and trigger auto-update if applicable."""
        logger.debug(f"Toggling interest: {interest}")

        try:
            if not interest or interest not in self.available_interests:
                logger.warning(f"Invalid interest provided: {interest}")
                return

            # Toggle interest in list
            if interest in self.selected_interests:
                self.selected_interests = [i for i in self.selected_interests if i != interest]
                logger.debug(f"Removed interest: {interest}")
            else:
                self.selected_interests = self.selected_interests + [interest]
                logger.debug(f"Added interest: {interest}")

            logger.info(f"Interests updated: {len(self.selected_interests)} selected")

            # Auto-generate if we've generated recommendations before
            if self.has_generated_once and self.selected_subreddit:
                logger.debug("Auto-generating recommendations for interest change")
                self._current_task_id += 1
                current_task = self._current_task_id
                return FeedShiftState.auto_generate_recommendations_delayed(current_task)

        except Exception as e:
            logger.error(f"Error toggling interest {interest}: {e}")

    @rx.event
    def set_toxicity_strictness(self, value: Union[List[float], float]) -> Optional[Any]:
        """Set toxicity strictness level and trigger auto-update if applicable."""
        try:
            if isinstance(value, list):
                new_strictness = float(value[0]) if value else DefaultValues.TOXICITY_STRICTNESS
            else:
                new_strictness = float(value)

            new_strictness = max(DefaultValues.TOXICITY_MIN, min(DefaultValues.TOXICITY_MAX, new_strictness))
            logger.debug(f"Setting toxicity strictness: {self.toxicity_strictness} -> {new_strictness}")
            self.toxicity_strictness = new_strictness

            if self.has_generated_once and self.selected_subreddit:
                logger.debug("Auto-generating recommendations for toxicity change")
                self._current_task_id += 1
                current_task = self._current_task_id
                return FeedShiftState.auto_generate_recommendations_delayed(current_task)

        except (ValueError, TypeError) as e:
            logger.error(f"Error setting toxicity strictness with value {value}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error setting toxicity strictness: {e}")

    @rx.event
    def set_diversity_strength(self, value: Union[List[float], float]) -> Optional[Any]:
        """Set diversity strength value and trigger auto-update if applicable."""
        try:
            if isinstance(value, list):
                new_diversity = float(value[0]) if value else DefaultValues.DIVERSITY_STRENGTH
            else:
                new_diversity = float(value)

            new_diversity = max(DefaultValues.DIVERSITY_MIN, min(DefaultValues.DIVERSITY_MAX, new_diversity))
            logger.debug(f"Setting diversity strength: {self.diversity_strength} -> {new_diversity}")
            self.diversity_strength = new_diversity

            if self.has_generated_once and self.selected_subreddit:
                logger.debug("Auto-generating recommendations for diversity change")
                self._current_task_id += 1
                current_task = self._current_task_id
                return FeedShiftState.auto_generate_recommendations_delayed(current_task)

        except (ValueError, TypeError) as e:
            logger.error(f"Error setting diversity strength with value {value}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error setting diversity strength: {e}")

    # === Recommendation Generation Event Handlers ===

    @rx.event
    def generate_recommendations(self) -> None:
        """Generate recommendations with current settings (manual trigger)."""
        if not self.selected_subreddit:
            logger.warning("Attempted to generate recommendations without subreddit")
            return

        logger.info(f"Manually generating recommendations for r/{self.selected_subreddit}")
        self.is_processing = True

        try:
            self.ranked_data = get_recommended_posts(
                self.selected_subreddit,
                self.selected_interests,
                self.toxicity_strictness,
                self.diversity_strength,
            )

            self.ranked_posts = self.ranked_data.to_dict("records")
            self._last_processed_value = self.toxicity_strictness
            self._last_processed_diversity = self.diversity_strength
            self.has_generated_once = True

            logger.info(f"Successfully generated {len(self.ranked_posts)} recommendations")

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        finally:
            self.is_processing = False

    @rx.event
    def auto_generate_recommendations(self, task_id: int) -> None:
        """Auto-generate recommendations immediately (for subreddit changes)."""
        logger.debug(f"Auto-generating recommendations (task {task_id})")

        if self._current_task_id != task_id:
            self._cancelled_tasks += 1
            logger.debug(f"Task {task_id} cancelled before execution")
            return

        if not self.selected_subreddit:
            logger.warning("No subreddit selected for auto-generation")
            return

        self.is_processing = True

        try:
            if self._current_task_id != task_id:
                self._cancelled_tasks += 1
                logger.debug(f"Task {task_id} cancelled during processing")
                return

            logger.info(f"Auto-generating recommendations for r/{self.selected_subreddit}")

            self.ranked_data = get_recommended_posts(
                self.selected_subreddit,
                self.selected_interests,
                self.toxicity_strictness,
                self.diversity_strength,
            )

            self.ranked_posts = self.ranked_data.to_dict("records")
            self._last_processed_value = self.toxicity_strictness
            self._last_processed_diversity = self.diversity_strength

            logger.info(f"Auto-generation completed: {len(self.ranked_posts)} recommendations")

        except Exception as e:
            logger.error(f"Error in auto-generation (task {task_id}): {e}")
        finally:
            self.is_processing = False

    @rx.event
    def auto_generate_recommendations_delayed(self, task_id: int) -> None:
        """Auto-generate recommendations with debouncing (for filter changes)."""
        logger.debug(f"Starting delayed auto-generation (task {task_id})")

        if self._current_task_id != task_id:
            self._cancelled_tasks += 1
            logger.debug(f"Task {task_id} cancelled after debounce")
            return

        if not self.selected_subreddit:
            logger.warning("No subreddit selected after debounce period")
            return

        if self._current_task_id != task_id:
            self._cancelled_tasks += 1
            logger.debug(f"Task {task_id} cancelled before processing")
            return

        self.is_processing = True

        try:
            if self._current_task_id != task_id:
                self._cancelled_tasks += 1
                logger.debug(f"Task {task_id} cancelled during processing")
                return

            logger.info(
                f"Auto-updating recommendations (task {task_id}) - "
                f"toxicity: {self.toxicity_strictness}, diversity: {self.diversity_strength}"
            )

            self.ranked_data = get_recommended_posts(
                self.selected_subreddit,
                self.selected_interests,
                self.toxicity_strictness,
                self.diversity_strength,
            )

            self.ranked_posts = self.ranked_data.to_dict("records")
            self._last_processed_value = self.toxicity_strictness
            self._last_processed_diversity = self.diversity_strength

            logger.info(f"Auto-update completed: {len(self.ranked_posts)} recommendations")

        except Exception as e:
            logger.error(f"Error in delayed auto-generation (task {task_id}): {e}")
        finally:
            self.is_processing = False

    @rx.event
    def reset_all_selections(self) -> None:
        """Reset all selections and results to initial state."""
        logger.info("Resetting all selections by user request")
        self.reset_to_initial_state()
        logger.info("All selections reset successfully")

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state for debugging and logging."""
        return {
            "selected_subreddit": self.selected_subreddit,
            "selected_interests_count": len(self.selected_interests),
            "toxicity_strictness": self.toxicity_strictness,
            "diversity_strength": self.diversity_strength,
            "has_results": len(self.ranked_posts) > 0,
            "is_processing": self.is_processing,
            "has_generated_once": self.has_generated_once,
            "current_task_id": self._current_task_id,
            "cancelled_tasks": self._cancelled_tasks,
        }
