"""
Dashboard constants and configuration for the FeedShift Reflex web interface.

This module defines UI-related constants, styling configurations, and enums used
throughout the dashboard application for consistent theming, sizing, and behavior.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import List, Tuple

from src.config.paths import DATA_DIR
from src.data.constants import DataCols, RedditDataCols

# Configure logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UIPath:
    """
    File paths used by the dashboard UI for data operations.

    Centralizes all file path configurations to ensure consistency
    across the dashboard application.

    Attributes:
        CSV_UPLOAD_PATH: Path for uploaded CSV files in dashboard
    """

    CSV_UPLOAD_PATH: Path = DATA_DIR / "dashboard" / "feedshift_data.csv"


@dataclass(frozen=True)
class InterestCategories:
    """
    Available interest categories for content filtering and personalization.

    Defines the complete set of interest categories that users can select
    to personalize their content recommendations. These categories are used
    for semantic matching with post content.

    Attributes:
        TECHNOLOGY: Technology and programming related content
        SPORTS: Sports, athletics, and competition content
        ENTERTAINMENT: Movies, TV, celebrities, and entertainment
        POLITICS: Political news, discussions, and analysis
        SCIENCE: Scientific research, discoveries, and education
        HEALTH: Health, wellness, and medical information
        BUSINESS: Business news, entrepreneurship, and finance
        TRAVEL: Travel destinations, tips, and experiences
        FOOD: Cooking, restaurants, and culinary content
        FASHION: Fashion trends, style, and clothing
    """

    TECHNOLOGY: str = "Technology"
    SPORTS: str = "Sports"
    ENTERTAINMENT: str = "Entertainment"
    POLITICS: str = "Politics"
    SCIENCE: str = "Science"
    HEALTH: str = "Health"
    BUSINESS: str = "Business"
    TRAVEL: str = "Travel"
    FOOD: str = "Food"
    FASHION: str = "Fashion"

    @classmethod
    def get_all(cls) -> List[str]:
        """
        Return all available interest categories as a list.

        Provides a programmatic way to access all interest categories
        for UI generation and validation purposes.

        Returns:
            List[str]: Complete list of all available interest categories

        Example:
            >>> categories = InterestCategories.get_all()
            >>> print(len(categories))  # 10
        """
        categories = [
            cls.TECHNOLOGY,
            cls.SPORTS,
            cls.ENTERTAINMENT,
            cls.POLITICS,
            cls.SCIENCE,
            cls.HEALTH,
            cls.BUSINESS,
            cls.TRAVEL,
            cls.FOOD,
            cls.FASHION,
        ]

        logger.debug(f"Retrieved {len(categories)} interest categories")
        return categories

    @classmethod
    def get_by_category_type(cls) -> dict[str, List[str]]:
        """
        Group interests by category type for organized display.

        Returns:
            dict[str, List[str]]: Dictionary mapping category types to interest lists
        """
        return {
            "information": [cls.TECHNOLOGY, cls.SCIENCE, cls.POLITICS, cls.BUSINESS],
            "lifestyle": [cls.HEALTH, cls.TRAVEL, cls.FOOD, cls.FASHION],
            "entertainment": [cls.ENTERTAINMENT, cls.SPORTS],
        }


@dataclass(frozen=True)
class UIConstants:
    """
    UI-related constants for consistent sizing, timing, and behavior.

    Centralizes all UI configuration values to ensure consistent behavior
    across the dashboard components and enable easy theme adjustments.

    Attributes:
        MAX_FILES: Maximum number of files allowed in upload
        UPLOAD_ZONE_MIN_HEIGHT: Minimum height for file upload zone
        UPLOAD_ZONE_MAX_WIDTH: Maximum width for file upload zone
        DEBOUNCE_DELAY: Delay in seconds for debouncing user input
        RESULTS_HEIGHT: Height of results scrollable area
        TOP_POSTS_LIMIT: Maximum number of posts to display
        CONTAINER_MAX_WIDTH: Maximum width for main containers
        CARD_MAX_WIDTH: Maximum width for content cards
        CONTROL_MAX_WIDTH: Maximum width for control panels
        LOGO_HEIGHT: Height of the application logo
        LOGO_FILENAME: Filename of the logo image
        DATE_TIME_FORMAT: Format string for timestamps
    """

    # File upload configuration
    MAX_FILES: int = 1
    UPLOAD_ZONE_MIN_HEIGHT: str = "150px"
    UPLOAD_ZONE_MAX_WIDTH: str = "400px"

    # User interaction timing
    DEBOUNCE_DELAY: float = 0.3  # 300ms debounce for responsive filtering

    # Content display configuration
    RESULTS_HEIGHT: str = "400px"
    TOP_POSTS_LIMIT: int = 10  # Number of top posts to display

    # Layout sizing
    CONTAINER_MAX_WIDTH: str = "1200px"
    CARD_MAX_WIDTH: str = "800px"
    CONTROL_MAX_WIDTH: str = "500px"
    SIDEBAR_WIDTH: str = "400px"
    SIDEBAR_MIN_WIDTH: str = "400px"

    # Branding
    LOGO_HEIGHT: str = "250px"
    LOGO_HEIGHT_SMALL: str = "200px"  # For results view
    LOGO_FILENAME: str = "feedshift_main_logo-removebg.png"

    # Formatting
    DATE_TIME_FORMAT: str = "%Y%m%d_%H%M%S"

    # Animation and transitions
    HOVER_TRANSITION_DURATION: str = "0.2s"
    CARD_HOVER_TRANSFORM: str = "translateY(-2px)"


@dataclass(frozen=True)
class DefaultValues:
    """
    Default values for various dashboard parameters and controls.

    Defines sensible defaults for all user-configurable parameters
    to ensure the application starts with reasonable settings.

    Attributes:
        TOXICITY_STRICTNESS: Default toxicity filtering level
        TOXICITY_MIN: Minimum allowed toxicity value
        TOXICITY_MAX: Maximum allowed toxicity value
        TOXICITY_STEP: Step size for toxicity slider
        DIVERSITY_STRENGTH: Default diversity strength
        DIVERSITY_MIN: Minimum diversity value
        DIVERSITY_MAX: Maximum diversity value
        DIVERSITY_STEP: Step size for diversity slider
    """

    # Toxicity filter defaults
    TOXICITY_STRICTNESS: float = 0.5
    TOXICITY_MIN: float = 0.0
    TOXICITY_MAX: float = 1.0
    TOXICITY_STEP: float = 0.05

    # Diversity filter defaults
    DIVERSITY_STRENGTH: float = 0.5
    DIVERSITY_MIN: float = 0.0
    DIVERSITY_MAX: float = 1.0
    DIVERSITY_STEP: float = 0.01


@dataclass(frozen=True)
class ContentFields:
    """
    Field names and fallback values for content display across different data sources.

    Provides a centralized mapping of field names used to extract content
    and metadata from different platform data structures, with fallback values
    for missing or invalid data.

    Attributes:
        CONTENT_FIELDS: Tuple of field names to check for main content
        AUTHOR_FIELDS: Tuple of field names to check for author information
        FALLBACK_CONTENT: Default text when no content is available
        FALLBACK_AUTHOR: Default text when no author is available
    """

    CONTENT_FIELDS: Tuple[str, ...] = (DataCols.PROCESSED_TEXT, RedditDataCols.PROCESSED_TITLE)
    AUTHOR_FIELDS: Tuple[str, ...] = (DataCols.AUTHOR, DataCols.USER, DataCols.USERNAME)
    FALLBACK_CONTENT: str = "No content available"
    FALLBACK_AUTHOR: str = "Unknown"

    # Additional metadata fields
    TIMESTAMP_FIELDS: Tuple[str, ...] = (DataCols.TIMESTAMP, RedditDataCols.TIMESTAMP)
    SCORE_FIELDS: Tuple[str, ...] = (DataCols.RECOMMENDATION_SCORE, RedditDataCols.SCORE)


@dataclass(frozen=True)
class ColorScheme:
    """
    Color scheme constants for consistent theming throughout the dashboard.

    Defines a comprehensive color palette using CSS custom properties
    for consistent theming and easy theme switching. Colors are organized
    by semantic meaning and usage context.

    Attributes:
        PRIMARY: Primary theme color (purple)
        SECONDARY: Secondary theme color (gray)
        Various CSS custom property references for specific shades
    """

    # Base color schemes
    PRIMARY: str = "purple"
    SECONDARY: str = "gray"

    # Success, warning, and error colors
    SUCCESS: str = "green"
    WARNING: str = "orange"
    ERROR: str = "red"

    # Purple palette (CSS custom properties)
    PURPLE_1: str = "var(--purple-1)"  # Lightest purple background
    PURPLE_2: str = "var(--purple-2)"  # Very light purple
    PURPLE_3: str = "var(--purple-3)"  # Light purple
    PURPLE_6: str = "var(--purple-6)"  # Medium purple
    PURPLE_7: str = "var(--purple-7)"  # Medium-dark purple
    PURPLE_8: str = "var(--purple-8)"  # Dark purple
    PURPLE_9: str = "var(--purple-9)"  # Primary purple
    PURPLE_11: str = "var(--purple-11)"  # Text purple

    # Gray palette (CSS custom properties)
    GRAY_9: str = "var(--gray-9)"  # Medium gray
    GRAY_10: str = "var(--gray-10)"  # Medium-dark gray
    GRAY_11: str = "var(--gray-11)"  # Dark gray
    GRAY_12: str = "var(--gray-12)"  # Darkest gray (main text)

    # Semantic color mappings
    BACKGROUND_PRIMARY: str = PURPLE_1
    BACKGROUND_SECONDARY: str = PURPLE_2
    BORDER_PRIMARY: str = PURPLE_6
    BORDER_SECONDARY: str = PURPLE_7
    TEXT_PRIMARY: str = GRAY_12
    TEXT_SECONDARY: str = GRAY_11
    TEXT_MUTED: str = GRAY_10
    ACCENT: str = PURPLE_9


class PostSimilarityType(StrEnum):
    """
    Enumeration of post similarity types for diversity control visualization.

    Provides human-readable labels for different levels of content similarity
    used in the diversity control slider. These types help users understand
    how the diversity setting affects content variety.

    Values:
        SAME: Posts are very similar (low diversity)
        SIMILAR: Posts have high similarity
        NEAR: Posts have moderate similarity
        DIVERSE: Posts have low similarity (high diversity)
        DIFFERENT: Posts are very different (maximum diversity)
    """

    SAME = "Same"
    SIMILAR = "Similar"
    NEAR = "Near"
    DIVERSE = "Diverse"
    DIFFERENT = "Different"

    @classmethod
    def get_description(cls, similarity_type: str) -> str:
        """
        Get a description for a similarity type.

        Args:
            similarity_type (str): The similarity type to describe

        Returns:
            str: Human-readable description of the similarity type
        """
        descriptions = {
            cls.SAME: "Content will be very similar to your interests",
            cls.SIMILAR: "Content will be quite similar with some variation",
            cls.NEAR: "Content will have moderate variety",
            cls.DIVERSE: "Content will be diverse with good variety",
            cls.DIFFERENT: "Content will be maximally diverse and different",
        }
        return descriptions.get(similarity_type, "Unknown similarity type")


# Performance and caching constants
@dataclass(frozen=True)
class PerformanceConstants:
    """
    Performance-related constants for optimization and caching.

    Attributes:
        CACHE_TTL: Time-to-live for cached responses
        MAX_CONCURRENT_REQUESTS: Maximum concurrent processing requests
        REQUEST_TIMEOUT: Timeout for individual requests
        BATCH_PROCESSING_SIZE: Size of batches for bulk operations
    """

    CACHE_TTL: int = 300  # 5 minutes
    MAX_CONCURRENT_REQUESTS: int = 5
    REQUEST_TIMEOUT: int = 30  # seconds
    BATCH_PROCESSING_SIZE: int = 100


# Validation constants
@dataclass(frozen=True)
class ValidationConstants:
    """
    Constants for input validation and limits.

    Attributes:
        MIN_INTERESTS: Minimum number of interests that can be selected
        MAX_INTERESTS: Maximum number of interests that can be selected
        MIN_CONTENT_LENGTH: Minimum content length for display
        MAX_CONTENT_LENGTH: Maximum content length for display
    """

    MIN_INTERESTS: int = 0
    MAX_INTERESTS: int = 10
    MIN_CONTENT_LENGTH: int = 1
    MAX_CONTENT_LENGTH: int = 500
    MIN_SUBREDDIT_NAME_LENGTH: int = 1
    MAX_SUBREDDIT_NAME_LENGTH: int = 21  # Reddit's max subreddit name length


# Error messages
@dataclass(frozen=True)
class ErrorMessages:
    """
    Centralized error messages for consistent user feedback.

    Attributes:
        NO_SUBREDDIT_SELECTED: Error when no subreddit is selected
        PROCESSING_FAILED: General processing failure message
        NO_RESULTS_FOUND: Message when no results are available
        NETWORK_ERROR: Network connectivity error
        INVALID_INPUT: Invalid input error
    """

    NO_SUBREDDIT_SELECTED: str = "Please select a subreddit to get recommendations"
    PROCESSING_FAILED: str = "Failed to generate recommendations. Please try again."
    NO_RESULTS_FOUND: str = "No posts found matching your criteria. Try adjusting your filters."
    NETWORK_ERROR: str = "Network error occurred. Please check your connection and try again."
    INVALID_INPUT: str = "Invalid input provided. Please check your settings."
    RATE_LIMIT_EXCEEDED: str = "Too many requests. Please wait a moment and try again."


# Success messages
@dataclass(frozen=True)
class SuccessMessages:
    """
    Success messages for positive user feedback.

    Attributes:
        RECOMMENDATIONS_GENERATED: Success message for recommendations
        SETTINGS_SAVED: Settings save confirmation
        DATA_LOADED: Data loading confirmation
    """

    RECOMMENDATIONS_GENERATED: str = "Recommendations generated successfully!"
    SETTINGS_SAVED: str = "Settings saved successfully"
    DATA_LOADED: str = "Data loaded successfully"
    FILTERS_APPLIED: str = "Filters applied and results updated"


def get_similarity_type_from_value(diversity_value: float) -> str:
    """
    Convert diversity strength value to similarity type enum.

    Args:
        diversity_value (float): Diversity strength value (0.0-1.0)

    Returns:
        PostSimilarityType: Corresponding similarity type

    Example:
        >>> similarity_type = get_similarity_type_from_value(0.3)
        >>> print(similarity_type)  # PostSimilarityType.DIFFERENT
    """
    if diversity_value < 0.4:
        return PostSimilarityType.DIFFERENT
    elif diversity_value < 0.6:
        return PostSimilarityType.DIVERSE
    elif diversity_value < 0.8:
        return PostSimilarityType.NEAR
    elif diversity_value < 1.0:
        return PostSimilarityType.SIMILAR
    else:
        return PostSimilarityType.SAME


def validate_ui_constants() -> bool:
    """
    Validate UI constants for consistency and correctness.

    Returns:
        bool: True if all constants are valid

    Raises:
        ValueError: If any constants are invalid
    """
    logger.debug("Validating UI constants")

    # Validate numeric ranges
    if not (0.0 <= DefaultValues.TOXICITY_STRICTNESS <= 1.0):
        raise ValueError("TOXICITY_STRICTNESS must be between 0.0 and 1.0")

    if not (0.0 <= DefaultValues.DIVERSITY_STRENGTH <= 1.0):
        raise ValueError("DIVERSITY_STRENGTH must be between 0.0 and 1.0")

    if UIConstants.TOP_POSTS_LIMIT <= 0:
        raise ValueError("TOP_POSTS_LIMIT must be positive")

    if UIConstants.DEBOUNCE_DELAY < 0:
        raise ValueError("DEBOUNCE_DELAY must be non-negative")

    logger.debug("UI constants validation passed")
    return True


# Initialize validation on module import
try:
    validate_ui_constants()
    logger.info("Dashboard constants initialized and validated successfully")
except Exception as e:
    logger.error(f"Dashboard constants validation failed: {e}")
    raise
