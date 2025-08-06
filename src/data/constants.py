"""
Data-related constants and column definitions for the FeedShift application.

This module defines column names, subreddit lists, and configuration constants
used throughout the data processing pipeline for various social media platforms.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, List

# Configure logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataCols:
    """
    Base column names for data processing across all platforms.

    Defines standardized column names used in DataFrames throughout the application.
    These serve as the foundation for platform-specific column definitions.

    Attributes:
        AUTHOR: Column name for content author/creator
        USER: Column name for user identifier
        USERNAME: Column name for display username
        TEXT: Column name for main text content
        TIMESTAMP: Column name for creation timestamp
        PROCESSED_TEXT: Column name for cleaned/processed text
        RECOMMENDATION_SCORE: Column name for overall recommendation score
        UNIQUENESS_SCORE: Column name for content uniqueness metric
        FRESHNESS_SCORE: Column name for content recency metric
        TOXICITY_SCORE: Column name for toxicity assessment
        DIVERSITY_SCORE: Column name for content diversity metric
        INTERESTS_SCORE: Column name for user interest alignment
        HASH: Column name for content hash/identifier
    """

    AUTHOR: str = "author"
    USER: str = "user"
    USERNAME: str = "username"
    TEXT: str = "text"
    TIMESTAMP: str = "timestamp"
    PROCESSED_TEXT: str = "processed_text"
    RECOMMENDATION_SCORE: str = "recommendation_score"
    UNIQUENESS_SCORE: str = "uniqueness_score"
    FRESHNESS_SCORE: str = "freshness_score"
    TOXICITY_SCORE: str = "toxicity_score"
    DIVERSITY_SCORE: str = "diversity_score"
    INTERESTS_SCORE: str = "interests_score"
    HASH: str = "hash"


@dataclass(frozen=True)
class RedditDataCols(DataCols):
    """
    Reddit-specific column names extending base data columns.

    Defines column names specific to Reddit's data structure, including
    post metadata, engagement metrics, and Reddit-specific attributes.

    Inherits all base columns from DataCols and adds Reddit-specific fields.

    Attributes:
        ID: Reddit post unique identifier
        IS_TEXT: Whether post is text-only (self post)
        TITLE: Post title text
        SCORE: Reddit post score (upvotes - downvotes)
        UPVOTE_RATIO: Ratio of upvotes to total votes
        NUM_COMMENTS: Number of comments on the post
        SUBREDDIT: Subreddit name where post was made
        SUBSCRIBERS: Number of subreddit subscribers
        NUM_CROSSPOSTS: Number of times post was crossposted
        POST_TYPE: Type of Reddit post (text, link, image, etc.)
        IS_NSFW: Whether post is marked as NSFW
        IS_BOT: Whether post was made by a bot account
        IS_MEGATHREAD: Whether post is a megathread
        PROCESSED_TITLE: Cleaned version of post title
    """

    ID: str = "id"
    AUTHOR: str = "author"  # Override base class for clarity
    IS_TEXT: str = "is_self"
    TITLE: str = "title"
    SCORE: str = "score"
    UPVOTE_RATIO: str = "upvote_ratio"
    NUM_COMMENTS: str = "num_comments"
    TIMESTAMP: str = "created_utc"  # Override base class for Reddit's UTC format
    SUBREDDIT: str = "subreddit"
    SUBSCRIBERS: str = "subscribers"
    NUM_CROSSPOSTS: str = "num_crossposts"
    POST_TYPE: str = "post_type"
    IS_NSFW: str = "over_18"
    IS_BOT: str = "is_bot"
    IS_MEGATHREAD: str = "is_megathread"
    TEXT: str = "selftext"  # Override base class for Reddit's field name
    PROCESSED_TITLE: str = "processed_title"


@dataclass(frozen=True)
class SubRedditList:
    """
    Predefined list of popular subreddits for content ingestion.

    Contains commonly used subreddit names organized by category/topic.
    These serve as default options for feed generation and can be extended
    based on user preferences or platform requirements.

    All subreddit names are stored without the 'r/' prefix for consistency.
    """

    # Entertainment & Humor
    ANIME: str = "anime"
    FUNNY: str = "funny"
    JOKES: str = "jokes"
    MEMES: str = "memes"
    MOVIES: str = "movies"

    # Lifestyle & Interests
    BOOKS: str = "books"
    FOOD: str = "food"
    GAMING: str = "gaming"
    SPORTS: str = "sports"

    # Information & Discussion
    NEWS: str = "news"
    SCIENCE: str = "science"
    TECHNOLOGY: str = "technology"
    RELATIONSHIP_ADVICE: str = "relationship_advice"

    @classmethod
    def get_all(cls) -> List[str]:
        """
        Return all available subreddit names as a list.

        Returns:
            List[str]: Complete list of all predefined subreddit names

        Example:
            >>> subreddits = SubRedditList.get_all()
            >>> print(len(subreddits))  # 13
        """
        subreddits = [
            cls.ANIME,
            cls.BOOKS,
            cls.FUNNY,
            cls.FOOD,
            cls.GAMING,
            cls.JOKES,
            cls.MEMES,
            cls.MOVIES,
            cls.NEWS,
            cls.RELATIONSHIP_ADVICE,
            cls.SCIENCE,
            cls.SPORTS,
            cls.TECHNOLOGY,
        ]

        logger.debug(f"Retrieved {len(subreddits)} predefined subreddits")
        return subreddits

    @classmethod
    def get_categories(cls) -> Dict[str, List[str]]:
        """
        Return subreddits organized by category.

        Returns:
            dict[str, List[str]]: Dictionary mapping category names to subreddit lists
        """
        return {
            "entertainment": [cls.ANIME, cls.FUNNY, cls.JOKES, cls.MEMES, cls.MOVIES],
            "lifestyle": [cls.BOOKS, cls.FOOD, cls.GAMING, cls.SPORTS],
            "information": [cls.NEWS, cls.SCIENCE, cls.TECHNOLOGY, cls.RELATIONSHIP_ADVICE],
        }


@dataclass(frozen=True)
class RedditConstant:
    """
    Default configuration constants for Reddit data ingestion.

    Defines sensible defaults for Reddit API parameters that can be
    overridden when needed for specific use cases.

    Attributes:
        LIMIT: Default number of posts to fetch per request
        TIME_FILTER: Default time period for 'top' post filtering
    """

    LIMIT: int = 10
    TIME_FILTER: str = "day"  # Options: hour, day, week, month, year, all


@dataclass(frozen=True)
class RedditDBConstants:
    """
    Database configuration constants for Reddit data storage.

    Defines database and table names used for Reddit data persistence.

    Attributes:
        DB_NAME: SQLite database filename for Reddit data
        REDDIT_TABLE: Table name for storing Reddit posts
    """

    DB_NAME: str = "reddit_posts.db"
    REDDIT_TABLE: str = "reddit_posts_table"


@dataclass(frozen=True)
class RedditIngestorConstants:
    """
    Configuration constants for Reddit API integration via PRAW.

    Defines PRAW (Python Reddit API Wrapper) specific configuration
    values used for Reddit API authentication and requests.

    Attributes:
        PRAW_SITE_NAME: PRAW configuration site name
        PRAW_USER_AGENT: User agent string for Reddit API requests
    """

    PRAW_SITE_NAME: str = "DEFAULT"
    PRAW_USER_AGENT: str = "FeedShift:v1.0.0 (by /u/feedshift_bot)"  # More descriptive user agent


class SortBy(StrEnum):
    """
    Enumeration of available Reddit post sorting methods.

    Defines the sorting options available for Reddit post retrieval,
    each providing different content discovery patterns.

    Values:
        HOT: Posts that are currently popular and active
        TOP: Highest-rated posts within a time period
        NEW: Most recently posted content
        RISING: Posts gaining popularity quickly
    """

    HOT = "hot"
    TOP = "top"
    NEW = "new"
    RISING = "rising"


# Global database configuration
DEFAULT_DATABASE: str = "feed_shift.db"
"""Default database filename for the FeedShift application."""

# Validation constants
MAX_SUBREDDIT_NAME_LENGTH: int = 50
"""Maximum allowed length for subreddit names."""

MIN_FETCH_LIMIT: int = 1
"""Minimum number of posts that can be fetched in a single request."""

MAX_FETCH_LIMIT: int = 1000
"""Maximum number of posts that can be fetched in a single request."""
