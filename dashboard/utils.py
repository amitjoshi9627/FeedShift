"""
FeedShift Dashboard Utilities

Contains utility functions, helpers, and business logic used
throughout the dashboard application.
"""

import logging
from typing import List

import pandas as pd

from dashboard.constants import PostSimilarityType, UIConstants
from src.engine.constants import Platform
from src.engine.engine_factory import get_engine

logger = logging.getLogger(__name__)


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
        subreddit: Name of the subreddit to fetch posts from
        selected_interests: List of user interest categories
        toxicity_strictness: Toxicity filtering strength (0.0-1.0)
        diversity_strength: Content diversity strength (0.0-1.0)

    Returns:
        DataFrame containing ranked posts with recommendation scores

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


def get_similarity_type_from_value(diversity_strength: float) -> str:
    """
    Convert numeric diversity strength to human-readable similarity type.

    Args:
        diversity_strength: Numeric diversity strength value (0.0-1.0)

    Returns:
        Human-readable similarity type string
    """
    if diversity_strength <= 0.2:
        return PostSimilarityType.SAME
    elif diversity_strength <= 0.4:
        return PostSimilarityType.SIMILAR
    elif diversity_strength <= 0.6:
        return PostSimilarityType.NEAR
    elif diversity_strength <= 0.8:
        return PostSimilarityType.DIVERSE
    else:
        return PostSimilarityType.DIFFERENT


def format_timestamp(timestamp: str) -> str:
    """
    Format timestamp for display in UI components.

    Args:
        timestamp: Raw timestamp string

    Returns:
        Formatted timestamp string
    """
    # Add your timestamp formatting logic here
    return timestamp


def validate_subreddit_name(subreddit: str) -> bool:
    """
    Validate if a subreddit name is valid.

    Args:
        subreddit: Subreddit name to validate

    Returns:
        True if valid, False otherwise
    """
    if not subreddit or not subreddit.strip():
        return False

    # Add more validation logic as needed
    return True


def calculate_recommendation_score(post_data: dict, user_preferences: dict) -> float:
    """
    Calculate recommendation score for a post based on user preferences.

    Args:
        post_data: Dictionary containing post information
        user_preferences: Dictionary containing user preferences

    Returns:
        Calculated recommendation score
    """
    # Add your scoring logic here
    return 0.0


def filter_posts_by_toxicity(posts: pd.DataFrame, strictness: float) -> pd.DataFrame:
    """
    Filter posts based on toxicity strictness level.

    Args:
        posts: DataFrame containing posts
        strictness: Toxicity filtering strictness (0.0-1.0)

    Returns:
        Filtered DataFrame
    """
    # Add your toxicity filtering logic here
    return posts


def apply_diversity_filtering(posts: pd.DataFrame, strength: float) -> pd.DataFrame:
    """
    Apply diversity filtering to posts based on strength level.

    Args:
        posts: DataFrame containing posts
        strength: Diversity filtering strength (0.0-1.0)

    Returns:
        Filtered DataFrame with applied diversity
    """
    # Add your diversity filtering logic here
    return posts
