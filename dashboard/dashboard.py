"""
FeedShift Dashboard - Interactive web interface for personalized content recommendations.

This module implements a Reflex-based web dashboard that allows users to:
- Select subreddits for content sourcing
- Choose personal interests for content filtering
- Adjust toxicity and diversity parameters
- View ranked and personalized post recommendations
- Real-time updates with debounced parameter changes

The dashboard provides an intuitive interface for the FeedShift recommendation
engine with automatic updates and responsive design.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import reflex as rx

from dashboard.constants import (
    ColorScheme,
    ContentFields,
    DefaultValues,
    InterestCategories,
    PostSimilarityType,
    UIConstants,
    get_similarity_type_from_value,
)
from src.data.constants import DataCols, RedditDataCols, SubRedditList
from src.engine.constants import Platform
from src.engine.engine_factory import get_engine

# Configure logger
logger = logging.getLogger(__name__)


class FeedShiftState(rx.State):
    """
    Main state management class for the FeedShift dashboard.

    This class manages all application state including user selections,
    processing status, and recommendation results. It handles real-time
    updates with debouncing and provides reactive data for the UI components.

    Key Features:
    - Reactive state management with automatic UI updates
    - Debounced parameter changes for performance
    - Task cancellation for resource management
    - Automatic recommendation regeneration
    - Comprehensive error handling and logging

    Attributes:
        data: Raw data DataFrame from selected subreddit
        saved_data_path: Path to saved data file
        selected_subreddit: Currently selected subreddit name
        toxicity_strictness: Toxicity filtering strength (0.0-1.0)
        diversity_strength: Content diversity strength (0.0-1.0)
        ranked_data: Processed and ranked DataFrame
        ranked_posts: List of ranked posts for display
        subreddit_selected: Boolean flag for subreddit selection
        is_processing: Boolean flag for processing state
        has_generated_once: Flag for first-time generation tracking
        selected_interests: List of selected user interests
        available_interests: List of all available interests
        available_subreddits: List of all available subreddits
    """

    # Core data state
    data: Optional[pd.DataFrame] = None
    saved_data_path: Union[str, Path] = ""

    # User selections
    selected_subreddit: str = ""
    toxicity_strictness: float = DefaultValues.TOXICITY_STRICTNESS
    diversity_strength: float = DefaultValues.DIVERSITY_STRENGTH
    selected_interests: List[str] = []

    # Results state
    ranked_data: Optional[pd.DataFrame] = None
    ranked_posts: List[Dict[str, Any]] = []

    # UI state
    subreddit_selected: bool = False
    is_processing: bool = False
    has_generated_once: bool = False

    # Task management for debouncing and cancellation
    _current_task_id: int = 0
    _last_processed_value: float = DefaultValues.TOXICITY_STRICTNESS
    _last_processed_diversity: float = DefaultValues.DIVERSITY_STRENGTH
    _cancelled_tasks: int = 0

    # Available options
    available_interests: List[str] = InterestCategories.get_all()
    available_subreddits: List[str] = SubRedditList.get_all()

    def __init__(self, **kwargs) -> None:
        """
        Initialize the FeedShift state with default values.

        Sets up the initial state and logs the initialization for debugging.

        Args:
            **kwargs: Additional keyword arguments passed to parent class
        """
        logger.info("Initializing FeedShiftState")
        super().__init__(**kwargs)
        self.reset_to_initial_state()
        logger.debug("FeedShiftState initialized with default values")

    def reset_to_initial_state(self) -> None:
        """
        Reset all state variables to their initial default values.

        This method is called during initialization and when the user
        explicitly resets the application state. It ensures a clean
        starting point for new recommendation sessions.
        """
        logger.info("Resetting FeedShiftState to initial values")

        # Core data
        self.data = None
        self.saved_data_path = ""

        # User selections
        self.selected_subreddit = ""
        self.toxicity_strictness = DefaultValues.TOXICITY_STRICTNESS
        self.diversity_strength = DefaultValues.DIVERSITY_STRENGTH
        self.selected_interests = []

        # Results
        self.ranked_data = None
        self.ranked_posts = []

        # UI state
        self.subreddit_selected = False
        self.is_processing = False
        self.has_generated_once = False

        # Task management
        self._current_task_id = 0
        self._last_processed_value = DefaultValues.TOXICITY_STRICTNESS
        self._last_processed_diversity = DefaultValues.DIVERSITY_STRENGTH
        self._cancelled_tasks = 0

        logger.debug("State reset completed")

    @rx.var
    def diversity_label(self) -> str:
        """
        Get human-readable diversity label based on current diversity strength.

        Converts the numeric diversity strength value into a user-friendly
        label that describes the expected content variety.

        Returns:
            str: Human-readable diversity label (Same, Similar, Near, Diverse, Different)
        """
        similarity_type = get_similarity_type_from_value(self.diversity_strength)
        logger.debug(f"Diversity strength {self.diversity_strength} mapped to {similarity_type}")
        return similarity_type

    @rx.var
    def processing_info(self) -> str:
        """
        Get current processing information for display in UI.

        Provides a comprehensive summary of current processing parameters
        for user feedback and debugging purposes.

        Returns:
            str: Formatted processing information string
        """
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

    @rx.var
    def can_generate_recommendations(self) -> bool:
        """
        Check if recommendations can be generated with current state.

        Validates that all required parameters are set for recommendation
        generation, primarily checking for subreddit selection.

        Returns:
            bool: True if recommendations can be generated
        """
        can_generate = bool(self.selected_subreddit and self.selected_subreddit.strip())
        logger.debug(f"Can generate recommendations: {can_generate}")
        return can_generate

    @rx.event
    def set_subreddit(self, subreddit: str) -> Optional[Any]:
        """
        Set the selected subreddit and trigger auto-generation if applicable.

        Updates the selected subreddit and clears previous results. If the user
        has generated recommendations before, automatically triggers new
        recommendation generation with the new subreddit.

        Args:
            subreddit (str): Name of the subreddit to select (without 'r/' prefix)

        Returns:
            Optional[Any]: Auto-generation task if applicable
        """
        logger.info(f"Setting subreddit to: {subreddit}")

        try:
            # Validate subreddit name
            if not subreddit or not subreddit.strip():
                logger.warning("Empty subreddit name provided")
                return

            # Update state
            self.selected_subreddit = subreddit.strip()
            self.subreddit_selected = True

            # Clear previous results when changing subreddit
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
        """
        Toggle interest selection on/off and trigger auto-update if applicable.

        Adds or removes an interest from the selected interests list. If the user
        has generated recommendations before, triggers auto-update with debouncing.

        Args:
            interest (str): Interest category to toggle

        Returns:
            Optional[Any]: Auto-generation task if applicable
        """
        logger.debug(f"Toggling interest: {interest}")

        try:
            # Validate interest
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
        """
        Set toxicity strictness level and trigger auto-update if applicable.

        Updates the toxicity filtering strength. Handles both list and scalar inputs
        from different UI components. Triggers auto-update with debouncing if the user
        has generated recommendations before.

        Args:
            value (Union[List[float], float]): New toxicity strictness value

        Returns:
            Optional[Any]: Auto-generation task if applicable
        """
        try:
            # Handle different input formats
            if isinstance(value, list):
                new_strictness = float(value[0]) if value else DefaultValues.TOXICITY_STRICTNESS
            else:
                new_strictness = float(value)

            # Validate range
            new_strictness = max(DefaultValues.TOXICITY_MIN, min(DefaultValues.TOXICITY_MAX, new_strictness))

            logger.debug(f"Setting toxicity strictness: {self.toxicity_strictness} -> {new_strictness}")
            self.toxicity_strictness = new_strictness

            # Auto-generate if we've generated recommendations before
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
        """
        Set diversity strength value and trigger auto-update if applicable.

        Updates the content diversity strength. Handles both list and scalar inputs
        from different UI components. Triggers auto-update with debouncing if the user
        has generated recommendations before.

        Args:
            value (Union[List[float], float]): New diversity strength value

        Returns:
            Optional[Any]: Auto-generation task if applicable
        """
        try:
            # Handle different input formats
            if isinstance(value, list):
                new_diversity = float(value[0]) if value else DefaultValues.DIVERSITY_STRENGTH
            else:
                new_diversity = float(value)

            # Validate range
            new_diversity = max(DefaultValues.DIVERSITY_MIN, min(DefaultValues.DIVERSITY_MAX, new_diversity))

            logger.debug(f"Setting diversity strength: {self.diversity_strength} -> {new_diversity}")
            self.diversity_strength = new_diversity

            # Auto-generate if we've generated recommendations before
            if self.has_generated_once and self.selected_subreddit:
                logger.debug("Auto-generating recommendations for diversity change")
                self._current_task_id += 1
                current_task = self._current_task_id
                return FeedShiftState.auto_generate_recommendations_delayed(current_task)

        except (ValueError, TypeError) as e:
            logger.error(f"Error setting diversity strength with value {value}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error setting diversity strength: {e}")

    @rx.event
    def generate_recommendations(self) -> None:
        """
        Generate recommendations with current settings (manual trigger).

        This method is called when the user explicitly clicks the generate
        recommendations button. It performs the full recommendation pipeline
        and updates the UI with results.

        Raises:
            Exception: If recommendation generation fails
        """
        if not self.selected_subreddit:
            logger.warning("Attempted to generate recommendations without subreddit")
            return

        logger.info(f"Manually generating recommendations for r/{self.selected_subreddit}")
        self.is_processing = True

        try:
            # Generate recommendations
            self.ranked_data = get_recommended_posts(
                self.selected_subreddit,
                self.selected_interests,
                self.toxicity_strictness,
                self.diversity_strength,
            )

            # Convert to display format
            self.ranked_posts = self.ranked_data.to_dict("records")

            # Update tracking variables
            self._last_processed_value = self.toxicity_strictness
            self._last_processed_diversity = self.diversity_strength
            self.has_generated_once = True

            logger.info(f"Successfully generated {len(self.ranked_posts)} recommendations")

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            # Could set an error state here for UI feedback

        finally:
            self.is_processing = False

    @rx.event
    def auto_generate_recommendations(self, task_id: int) -> None:
        """
        Auto-generate recommendations immediately (for subreddit changes).

        This method is called when the subreddit changes and the user has
        previously generated recommendations. It provides immediate feedback
        without debouncing.

        Args:
            task_id (int): Unique task identifier for cancellation management
        """
        logger.debug(f"Auto-generating recommendations (task {task_id})")

        # Check if this task is still current
        if self._current_task_id != task_id:
            self._cancelled_tasks += 1
            logger.debug(f"Task {task_id} cancelled before execution")
            return

        if not self.selected_subreddit:
            logger.warning("No subreddit selected for auto-generation")
            return

        self.is_processing = True

        try:
            # Final check before expensive operation
            if self._current_task_id != task_id:
                self._cancelled_tasks += 1
                logger.debug(f"Task {task_id} cancelled during processing")
                return

            logger.info(f"Auto-generating recommendations for r/{self.selected_subreddit}")

            # Generate recommendations
            self.ranked_data = get_recommended_posts(
                self.selected_subreddit,
                self.selected_interests,
                self.toxicity_strictness,
                self.diversity_strength,
            )

            # Update display data
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
        """
        Auto-generate recommendations with debouncing (for filter changes).

        This method is called when filter parameters change. It includes
        debouncing to prevent excessive API calls when users are actively
        adjusting sliders or selecting multiple interests.

        Args:
            task_id (int): Unique task identifier for cancellation management
        """
        logger.debug(f"Starting delayed auto-generation (task {task_id})")

        # Note: Debouncing removed - immediate execution for now
        # Original: await asyncio.sleep(UIConstants.DEBOUNCE_DELAY)

        # Check if this task is still the current one
        if self._current_task_id != task_id:
            self._cancelled_tasks += 1
            logger.debug(f"Task {task_id} cancelled after debounce")
            return

        # Check if we still have subreddit selected
        if not self.selected_subreddit:
            logger.warning("No subreddit selected after debounce period")
            return

        # Double-check the task ID right before processing
        if self._current_task_id != task_id:
            self._cancelled_tasks += 1
            logger.debug(f"Task {task_id} cancelled before processing")
            return

        self.is_processing = True

        try:
            # Final check before expensive operation
            if self._current_task_id != task_id:
                self._cancelled_tasks += 1
                logger.debug(f"Task {task_id} cancelled during processing")
                return

            logger.info(
                f"Auto-updating recommendations (task {task_id}) - "
                f"toxicity: {self.toxicity_strictness}, diversity: {self.diversity_strength}"
            )

            # Generate recommendations
            self.ranked_data = get_recommended_posts(
                self.selected_subreddit,
                self.selected_interests,
                self.toxicity_strictness,
                self.diversity_strength,
            )

            # Update display data
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
        """
        Reset all selections and results to initial state.

        Provides a clean slate for users to start over with new selections.
        This is useful when users want to completely change their preferences.
        """
        logger.info("Resetting all selections by user request")
        self.reset_to_initial_state()
        logger.info("All selections reset successfully")

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current state for debugging and logging.

        Returns:
            Dict[str, Any]: Dictionary containing key state information
        """
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


def get_recommended_posts(
    subreddit: str,
    selected_interests: List[str],
    toxicity_strictness: float,
    diversity_strength: float,
) -> pd.DataFrame:
    """
    Get recommended posts using the FeedShift engine with specified parameters.

    This function interfaces with the core recommendation engine to fetch
    and rank posts from the specified subreddit based on user preferences.

    Args:
        subreddit (str): Name of the subreddit to fetch posts from
        selected_interests (List[str]): List of user interest categories
        toxicity_strictness (float): Toxicity filtering strength (0.0-1.0)
        diversity_strength (float): Content diversity strength (0.0-1.0)

    Returns:
        pd.DataFrame: DataFrame containing ranked posts with recommendation scores

    Raises:
        Exception: If engine creation or recommendation generation fails
    """
    logger.info(f"Getting recommendations for r/{subreddit}")
    logger.debug(
        f"Parameters: interests={len(selected_interests)}, "
        f"toxicity={toxicity_strictness}, diversity={diversity_strength}"
    )

    try:
        # Get engine for the platform and subreddit
        engine = get_engine(Platform.REDDIT, subreddit)
        logger.debug(f"Created engine for r/{subreddit}")

        # Run recommendation engine
        ranked_data = engine.run(
            interests=selected_interests, toxicity_strictness=toxicity_strictness, diversity_strength=diversity_strength
        ).head(UIConstants.TOP_POSTS_LIMIT)

        logger.info(f"Successfully generated {len(ranked_data)} recommendations")
        return ranked_data

    except Exception as e:
        logger.error(f"Error getting recommendations for r/{subreddit}: {e}")
        raise


# UI Component Functions with Enhanced Documentation


def subreddit_selector() -> rx.Component:
    """
    Create an enhanced subreddit selector component with grid layout.

    Provides a visually appealing grid of subreddit options with:
    - Visual selection indicators
    - Responsive grid layout
    - Hover effects and smooth transitions
    - Current selection display

    Returns:
        rx.Component: Configured subreddit selector card component
    """
    logger.debug("Creating subreddit selector component")

    return rx.card(
        rx.vstack(
            # Header section
            rx.hstack(
                rx.icon("reddit", size=20, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Select Subreddit",
                    font_size="1.15em",
                    font_weight="600",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
                margin_bottom="0.5rem",
            ),
            # Current selection display
            rx.text(
                "Selected: "
                + f"{rx.cond(FeedShiftState.selected_subreddit, f'r/{FeedShiftState.selected_subreddit}', 'None')}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
            # Subreddit grid
            rx.box(
                rx.foreach(
                    FeedShiftState.available_subreddits,
                    lambda subreddit: rx.button(
                        rx.hstack(
                            rx.cond(
                                FeedShiftState.selected_subreddit == subreddit,
                                rx.icon("check-circle", size=12),
                                rx.icon("circle", size=12),
                            ),
                            rx.text(f"r/{subreddit}", font_size="0.88em"),
                            spacing="1",
                            align="center",
                        ),
                        variant=rx.cond(
                            FeedShiftState.selected_subreddit == subreddit,
                            "solid",
                            "soft",
                        ),
                        color_scheme=rx.cond(
                            FeedShiftState.selected_subreddit == subreddit,
                            ColorScheme.PRIMARY,
                            ColorScheme.SECONDARY,
                        ),
                        size="2",
                        margin="0.15rem",
                        cursor="pointer",
                        on_click=FeedShiftState.set_subreddit(subreddit),
                        _hover={"cursor": "pointer"},
                        flex="1 1 auto",
                        min_width="0",
                        width="100%",
                    ),
                ),
                display="grid",
                grid_template_columns="repeat(auto-fit, minmax(140px, 1fr))",
                gap="0.5rem",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        margin_bottom="1.25rem",
        padding="1.25rem",
        box_shadow="0 4px 12px rgba(0,0,0,0.05)",
    )


def interest_selector() -> rx.Component:
    """
    Create an enhanced interest selector component with toggle functionality.

    Provides a multi-select interface for user interests with:
    - Visual selection indicators (check/plus icons)
    - Responsive grid layout
    - Selection counter
    - Smooth toggle animations

    Returns:
        rx.Component: Configured interest selector card component
    """
    logger.debug("Creating interest selector component")

    return rx.card(
        rx.vstack(
            # Header section
            rx.hstack(
                rx.icon("heart", size=20, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Your Interests",
                    font_size="1.15em",
                    font_weight="600",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
                margin_bottom="0.5rem",
            ),
            # Selection counter
            rx.text(
                f"Selected: {FeedShiftState.selected_interests.length()}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
            # Interest grid
            rx.box(
                rx.foreach(
                    FeedShiftState.available_interests,
                    lambda interest: rx.button(
                        rx.hstack(
                            rx.cond(
                                FeedShiftState.selected_interests.contains(interest),
                                rx.icon("check", size=12),
                                rx.icon("plus", size=12),
                            ),
                            rx.text(interest, font_size="0.88em"),
                            spacing="1",
                            align="center",
                        ),
                        variant=rx.cond(
                            FeedShiftState.selected_interests.contains(interest),
                            "solid",
                            "soft",
                        ),
                        color_scheme=rx.cond(
                            FeedShiftState.selected_interests.contains(interest),
                            ColorScheme.PRIMARY,
                            ColorScheme.SECONDARY,
                        ),
                        size="2",
                        margin="0.15rem",
                        cursor="pointer",
                        on_click=FeedShiftState.toggle_interest(interest),
                        _hover={"cursor": "pointer"},
                        flex="1 1 auto",
                        min_width="0",
                    ),
                ),
                display="grid",
                grid_template_columns="repeat(auto-fit, minmax(120px, 1fr))",
                gap="0.5rem",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        margin_bottom="1.25rem",
        padding="1.25rem",
        box_shadow="0 4px 12px rgba(0,0,0,0.05)",
    )


def toxicity_control() -> rx.Component:
    """
    Create an enhanced toxicity control slider with visual feedback.

    Provides a slider interface for toxicity filtering with:
    - Real-time value display
    - Descriptive labels (Lenient/Strict)
    - Smooth slider interaction
    - Visual feedback for current setting

    Returns:
        rx.Component: Configured toxicity control card component
    """
    logger.debug("Creating toxicity control component")

    return rx.card(
        rx.vstack(
            # Header section
            rx.hstack(
                rx.icon("shield-check", size=20, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Toxicity Filter",
                    font_size="1.15em",
                    font_weight="600",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
                margin_bottom="0.5rem",
            ),
            # Current value display
            rx.text(
                f"Strictness: {FeedShiftState.toxicity_strictness:.2f}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
            # Slider control
            rx.slider(
                default_value=[FeedShiftState.toxicity_strictness],
                min_=DefaultValues.TOXICITY_MIN,
                max=DefaultValues.TOXICITY_MAX,
                step=DefaultValues.TOXICITY_STEP,
                on_change=FeedShiftState.set_toxicity_strictness,
                color_scheme=ColorScheme.PRIMARY,
                size="2",
                cursor="pointer",
            ),
            # Descriptive labels
            rx.hstack(
                rx.text("Lenient", font_size="0.85em", color=ColorScheme.GRAY_10),
                rx.spacer(),
                rx.text("Strict", font_size="0.85em", color=ColorScheme.GRAY_10),
                width="100%",
                margin_top="0.5rem",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        margin_bottom="1.25rem",
        padding="1.25rem",
        box_shadow="0 4px 12px rgba(0,0,0,0.05)",
    )


def diversity_control() -> rx.Component:
    """
    Create an enhanced diversity control slider with semantic labels.

    Provides a slider interface for diversity control with:
    - Real-time similarity type display
    - Semantic labels (Different/Same)
    - Smooth slider interaction
    - Visual feedback for diversity level

    Returns:
        rx.Component: Configured diversity control card component
    """
    logger.debug("Creating diversity control component")

    return rx.card(
        rx.vstack(
            # Header section
            rx.hstack(
                rx.icon("shuffle", size=20, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Diversity Control",
                    font_size="1.15em",
                    font_weight="600",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
                margin_bottom="0.5rem",
            ),
            # Current value and label display
            rx.text(
                f"{FeedShiftState.diversity_label}: {FeedShiftState.diversity_strength:.2f}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
            # Slider control
            rx.slider(
                default_value=[FeedShiftState.diversity_strength],
                min_=DefaultValues.DIVERSITY_MIN,
                max=DefaultValues.DIVERSITY_MAX,
                step=DefaultValues.DIVERSITY_STEP,
                on_change=FeedShiftState.set_diversity_strength,
                color_scheme=ColorScheme.PRIMARY,
                size="2",
                cursor="pointer",
            ),
            # Semantic labels
            rx.hstack(
                rx.text(
                    PostSimilarityType.DIFFERENT,
                    font_size="0.85em",
                    color=ColorScheme.GRAY_10,
                ),
                rx.spacer(),
                rx.text(
                    PostSimilarityType.SAME,
                    font_size="0.85em",
                    color=ColorScheme.GRAY_10,
                ),
                width="100%",
                margin_top="0.5rem",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        margin_bottom="1.25rem",
        padding="1.25rem",
        box_shadow="0 4px 12px rgba(0,0,0,0.05)",
    )


def action_buttons() -> rx.Component:
    """
    Create action buttons for generating recommendations and resetting state.

    Provides control buttons with:
    - Dynamic button text based on state
    - Loading indicators during processing
    - Auto-update status indicator
    - Reset functionality

    Returns:
        rx.Component: Configured action buttons card component
    """
    logger.debug("Creating action buttons component")

    return rx.card(
        rx.vstack(
            # Generate/Regenerate button with dynamic content
            rx.button(
                rx.cond(
                    FeedShiftState.is_processing,
                    rx.hstack(
                        rx.spinner(size="3"),
                        rx.text("Generating..."),
                        spacing="2",
                        align="center",
                    ),
                    rx.cond(
                        FeedShiftState.has_generated_once,
                        rx.hstack(
                            rx.icon("refresh-cw", size=16),
                            rx.text("Regenerate"),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("zap", size=16),
                            rx.text("Get Recommendations"),
                            spacing="2",
                            align="center",
                        ),
                    ),
                ),
                on_click=FeedShiftState.generate_recommendations,
                color_scheme=ColorScheme.PRIMARY,
                size="3",
                width="100%",
                disabled=~FeedShiftState.can_generate_recommendations | FeedShiftState.is_processing,
                cursor="pointer",
                _hover={"cursor": "pointer"},
                padding="0.75rem 1.5rem",
                margin_bottom="0.75rem",
            ),
            # Auto-update indicator
            rx.cond(
                FeedShiftState.has_generated_once,
                rx.text(
                    "🔄 Auto-updates enabled",
                    font_size="0.85em",
                    color=ColorScheme.PURPLE_11,
                    text_align="center",
                    margin_bottom="0.5rem",
                    font_style="italic",
                ),
                rx.fragment(),
            ),
            # Reset button
            rx.button(
                rx.hstack(
                    rx.icon("refresh-ccw", size=14),
                    rx.text("Reset All Selections"),
                    spacing="2",
                    align="center",
                ),
                on_click=FeedShiftState.reset_all_selections,
                variant="soft",
                color_scheme=ColorScheme.SECONDARY,
                size="2",
                width="100%",
                cursor="pointer",
                _hover={"cursor": "pointer"},
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        padding="1.25rem",
        box_shadow="0 4px 12px rgba(0,0,0,0.05)",
        background=f"rgba({ColorScheme.PURPLE_1}, 0.5)",
        border=f"1px solid {ColorScheme.PURPLE_7}",
    )


def sidebar() -> rx.Component:
    """
    Create the main sidebar with all control components and scrolling capability.

    Provides a fixed sidebar containing:
    - All control components (subreddit, interests, filters, actions)
    - Scrollable content for long lists
    - Fixed positioning for consistent access
    - Themed styling and shadows

    Returns:
        rx.Component: Configured sidebar component
    """
    logger.debug("Creating main sidebar component")

    return rx.box(
        rx.scroll_area(
            rx.vstack(
                # Sidebar header
                rx.hstack(
                    rx.icon("settings", size=24, color=ColorScheme.PURPLE_9),
                    rx.text(
                        "Customize Feed",
                        font_size="1.3em",
                        font_weight="700",
                        color=ColorScheme.GRAY_12,
                    ),
                    align="center",
                    spacing="2",
                    margin_bottom="1.5rem",
                ),
                # Control components
                subreddit_selector(),
                interest_selector(),
                toxicity_control(),
                diversity_control(),
                action_buttons(),
                spacing="0",
                width="100%",
                padding="0 0.5rem 0 0",  # Right padding for scrollbar
            ),
            height="calc(100vh - 4rem)",  # Full viewport height minus top padding
            scrollbars="vertical",
            width="100%",
        ),
        width=UIConstants.SIDEBAR_WIDTH,
        min_width=UIConstants.SIDEBAR_MIN_WIDTH,
        height="calc(100vh - 4rem)",
        position="fixed",
        top="2rem",
        left="2rem",
        padding="1.5rem",
        background=f"rgba({ColorScheme.PURPLE_1}, 0.3)",
        border_radius="0 16px 16px 0",
        border=f"1px solid {ColorScheme.PURPLE_6}",
        box_shadow="0 6px 20px rgba(0,0,0,0.08)",
        z_index="100",
    )


def processing_overlay() -> rx.Component:
    """
    Create a processing overlay with status information.

    Shows processing status with:
    - Loading spinner
    - Dynamic status messages
    - Current processing parameters
    - Conditional display based on processing state

    Returns:
        rx.Component: Processing overlay component
    """
    logger.debug("Creating processing overlay component")

    return rx.cond(
        FeedShiftState.is_processing,
        rx.center(
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.spinner(size="3", color=ColorScheme.PURPLE_9),
                        rx.text(
                            rx.cond(
                                FeedShiftState.has_generated_once,
                                "Updating recommendations...",
                                "Generating recommendations...",
                            ),
                            font_weight="600",
                            color=ColorScheme.PURPLE_11,
                            font_size="1.1em",
                        ),
                        spacing="3",
                        align="center",
                    ),
                    rx.text(
                        FeedShiftState.processing_info,
                        font_size="0.95em",
                        color=ColorScheme.GRAY_11,
                        text_align="center",
                        margin_top="0.5rem",
                    ),
                    spacing="3",
                    align="center",
                ),
                padding="1.75rem 2.5rem",
                background=ColorScheme.PURPLE_2,
                border=f"1px solid {ColorScheme.PURPLE_7}",
                border_radius="14px",
            ),
            width="100%",
            margin="1.5rem 0",
        ),
        rx.fragment(),
    )


def post_card(post: Dict[str, Any], index: int) -> rx.Component:
    """
    Create an enhanced post card component with comprehensive content display.

    Displays post information with:
    - Rank badge
    - Content with overflow handling
    - Author and timestamp information
    - Recommendation score
    - Hover effects and responsive design

    Args:
        post (Dict[str, Any]): Post data dictionary
        index (int): Post ranking index

    Returns:
        rx.Component: Configured post card component
    """
    logger.debug(f"Creating post card for index {index}")

    # Extract content with fallback logic
    content = rx.cond(
        post.get(ContentFields.CONTENT_FIELDS[0]),
        post.get(ContentFields.CONTENT_FIELDS[0]),
        rx.cond(
            post.get(ContentFields.CONTENT_FIELDS[1]),
            post.get(ContentFields.CONTENT_FIELDS[1]),
            ContentFields.FALLBACK_CONTENT,
        ),
    )

    # Extract author with fallback logic
    author = rx.cond(
        post.get(ContentFields.AUTHOR_FIELDS[0]),
        post.get(ContentFields.AUTHOR_FIELDS[0]),
        rx.cond(
            post.get(ContentFields.AUTHOR_FIELDS[1]),
            post.get(ContentFields.AUTHOR_FIELDS[1]),
            rx.cond(
                post.get(ContentFields.AUTHOR_FIELDS[2]),
                post.get(ContentFields.AUTHOR_FIELDS[2]),
                ContentFields.FALLBACK_AUTHOR,
            ),
        ),
    )

    return rx.card(
        rx.vstack(
            # Header with rank and metadata
            rx.hstack(
                rx.badge(
                    index + 1,
                    color_scheme=ColorScheme.PRIMARY,
                    variant="soft",
                    padding="0.25rem 0.75rem",
                    font_size="0.9em",
                ),
                rx.spacer(),
                rx.text(
                    f"🕒 {post.get(RedditDataCols.TIMESTAMP)} • 🌐 r/{FeedShiftState.selected_subreddit}",
                    font_size="0.9em",
                    color=ColorScheme.GRAY_11,
                ),
                width="100%",
                align="center",
                margin_bottom="0.5rem",
            ),
            # Main content with overflow handling
            rx.text(
                content,
                color=ColorScheme.GRAY_12,
                line_height="1.6",
                font_size="0.95em",
                overflow="hidden",
                text_overflow="ellipsis",
                max_height="6em",
                display="-webkit-box",
                webkit_line_clamp="4",
                webkit_box_orient="vertical",
            ),
            # Footer with score and author
            rx.hstack(
                rx.text(
                    rx.cond(
                        post.get(DataCols.RECOMMENDATION_SCORE),
                        f"Score: {post.get(DataCols.RECOMMENDATION_SCORE)}",
                        "Score: N/A",
                    ),
                    font_weight="600",
                    color=ColorScheme.PURPLE_11,
                    font_size="0.9em",
                ),
                rx.spacer(),
                rx.text(
                    f"Author: {author}",
                    color=ColorScheme.GRAY_10,
                    font_size="0.85em",
                ),
                width="100%",
                align="center",
                margin_top="0.75rem",
            ),
            spacing="3",
            align="start",
        ),
        padding="1.75rem",
        margin="0.85rem 0",
        variant="surface",
        border_left=f"4px solid {ColorScheme.PURPLE_7}",
        width="100%",
        min_height="180px",
        _hover={
            "box_shadow": "0 6px 14px rgba(0,0,0,0.08)",
            "transform": UIConstants.CARD_HOVER_TRANSFORM,
            "transition": f"all {UIConstants.HOVER_TRANSITION_DURATION} ease",
        },
    )


def welcome_section() -> rx.Component:
    """
    Create the welcome section shown when no recommendations are generated yet.

    Provides an onboarding experience with:
    - Application branding and logo
    - Clear instructions for first-time users
    - Step-by-step usage guide
    - Engaging visual design

    Returns:
        rx.Component: Welcome section component
    """
    logger.debug("Creating welcome section component")

    return rx.center(
        rx.vstack(
            # Logo and branding
            rx.image(
                src=UIConstants.LOGO_FILENAME,
                alt="Feedshift Logo",
                height=UIConstants.LOGO_HEIGHT,
                object_fit="contain",
                margin_bottom="1.5rem",
            ),
            rx.heading(
                "FeedShift Recommendation Engine",
                size="5",
                color=ColorScheme.PURPLE_11,
                margin_bottom="1rem",
            ),
            rx.text(
                "Select a subreddit, "
                + "customize your preferences, and click 'Get Recommendations' to see personalized posts",
                color=ColorScheme.GRAY_11,
                font_size="1.05em",
                margin_bottom="2rem",
                text_align="center",
                max_width="600px",
            ),
            # Instructions card
            rx.card(
                rx.vstack(
                    rx.text(
                        "How to use:",
                        font_size="1.1em",
                        font_weight="600",
                        color=ColorScheme.GRAY_12,
                        margin_bottom="0.75rem",
                    ),
                    rx.vstack(
                        rx.hstack(
                            rx.icon("check-circle", size=16, color=ColorScheme.PURPLE_9),
                            rx.text("1. Choose a subreddit from the sidebar", color=ColorScheme.GRAY_11),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("check-circle", size=16, color=ColorScheme.PURPLE_9),
                            rx.text("2. Select your interests (optional)", color=ColorScheme.GRAY_11),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("check-circle", size=16, color=ColorScheme.PURPLE_9),
                            rx.text("3. Adjust toxicity and diversity filters", color=ColorScheme.GRAY_11),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("check-circle", size=16, color=ColorScheme.PURPLE_9),
                            rx.text("4. Click 'Get Recommendations' to see results", color=ColorScheme.GRAY_11),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("zap", size=16, color=ColorScheme.PURPLE_9),
                            rx.text(
                                "5. After first generation, filters auto-update results!",
                                color=ColorScheme.PURPLE_11,
                                font_weight="600",
                            ),
                            spacing="2",
                            align="center",
                        ),
                        spacing="1",
                        align="start",
                    ),
                    spacing="2",
                    align="start",
                ),
                padding="1.5rem",
                background=ColorScheme.PURPLE_2,
                border=f"1px solid {ColorScheme.PURPLE_6}",
                max_width="500px",
            ),
            align="center",
            spacing="2",
            width="100%",
        ),
        width="100%",
        flex="1",
        padding="2rem",
    )


def results_section() -> rx.Component:
    """
    Create the results section shown when recommendations are available.

    Displays recommendation results with:
    - Compact logo and branding
    - Results header with context
    - Processing overlay when updating
    - Scrollable list of post cards
    - Empty state handling

    Returns:
        rx.Component: Results section component
    """
    logger.debug("Creating results section component")

    return rx.vstack(
        # Compact logo and branding for results view
        rx.center(
            rx.vstack(
                rx.image(
                    src=UIConstants.LOGO_FILENAME,
                    alt="Feedshift Logo",
                    height=UIConstants.LOGO_HEIGHT_SMALL,
                    object_fit="contain",
                    margin_bottom="0.75rem",
                ),
                rx.text(
                    "FeedShift Recommendations",
                    font_size="1.2em",
                    font_weight="600",
                    color=ColorScheme.PURPLE_11,
                    margin_bottom="1.5rem",
                ),
                spacing="1",
                align="center",
            ),
            margin_bottom="1rem",
        ),
        # Main header section
        rx.hstack(
            rx.icon("trending-up", size=30, color=ColorScheme.PURPLE_9),
            rx.heading(
                f"Top Posts from r/{FeedShiftState.selected_subreddit}",
                size="6",
                color=ColorScheme.GRAY_12,
            ),
            align="center",
            spacing="3",
            justify="center",
        ),
        rx.text(
            "Showing top posts tailored to your preferences",
            color=ColorScheme.GRAY_11,
            font_size="1.05em",
            margin_bottom="1.25rem",
            text_align="center",
        ),
        # Processing overlay
        processing_overlay(),
        # Results content with conditional display
        rx.cond(
            FeedShiftState.ranked_posts.length() > 0,
            rx.cond(
                ~FeedShiftState.is_processing,
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(FeedShiftState.ranked_posts, post_card),
                        spacing="3",
                        width="100%",
                        align="center",
                    ),
                    height="65vh",  # Reduced to accommodate logo
                    scrollbars="vertical",
                    width="100%",
                    min_width="700px",
                ),
                rx.fragment(),
            ),
            rx.center(
                rx.text(
                    "No posts available. Try adjusting your filters or selecting different interests.",
                    color=ColorScheme.GRAY_11,
                    font_size="1.1em",
                    text_align="center",
                ),
                padding="2.5rem",
            ),
        ),
        spacing="2",
        width="100%",
        max_width=UIConstants.CARD_MAX_WIDTH,
        align="center",
    )


def main_content() -> rx.Component:
    """
    Create the main content area with conditional display logic.

    Provides the main content area that switches between:
    - Welcome section for first-time users
    - Results section when recommendations are available
    - Proper spacing and layout for sidebar accommodation

    Returns:
        rx.Component: Main content area component
    """
    logger.debug("Creating main content area component")

    return rx.box(
        rx.center(
            rx.cond(
                FeedShiftState.ranked_posts.length() > 0,
                results_section(),
                welcome_section(),
            ),
            width="100%",
        ),
        width="100%",
        flex="1",
        # Add left margin to account for fixed sidebar
        margin_left="140px",  # Adjusted for sidebar width
        padding_right="2rem",
    )


def index() -> rx.Component:
    """
    Create the main application index page with complete layout.

    Assembles the complete dashboard interface with:
    - Dark/light mode toggle
    - Fixed sidebar with all controls
    - Main content area with conditional display
    - Themed background with gradients
    - Responsive layout and proper spacing

    Returns:
        rx.Component: Complete dashboard application page
    """
    logger.info("Creating main dashboard index page")

    return rx.box(
        # Color mode toggle button
        rx.color_mode.button(
            position="fixed",
            top="1.25rem",
            right="1.25rem",
            z_index="999",
            size="3",
        ),
        # Fixed sidebar with all controls
        sidebar(),
        # Main content area
        main_content(),
        # Themed background
        background=f"radial-gradient(ellipse at top, {ColorScheme.PURPLE_3}, transparent), "
        + f"radial-gradient(ellipse at bottom, {ColorScheme.PURPLE_2}, transparent)",
        min_height="100vh",
        width="100%",
        padding_top="2rem",
    )


# Application Configuration and Initialization

# Configure Reflex application with theme and settings
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="large",
        accent_color=ColorScheme.PRIMARY,
    )
)

# Add the main page to the application
app.add_page(index)

logger.info("FeedShift dashboard application configured and ready to serve")
