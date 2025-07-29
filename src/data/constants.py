from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class DataCols:
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
    ID: str = "id"
    AUTHOR: str = "author"
    IS_TEXT: str = "is_self"
    TITLE: str = "title"
    SCORE: str = "score"
    UPVOTE_RATIO: str = "upvote_ratio"
    NUM_COMMENTS: str = "num_comments"
    TIMESTAMP: str = "created_utc"
    SUBREDDIT: str = "subreddit"
    SUBSCRIBERS: str = "subscribers"
    NUM_CROSSPOSTS: str = "num_crossposts"
    POST_TYPE: str = "post_type"
    IS_NSFW: str = "over_18"
    IS_BOT: str = "is_bot"
    IS_MEGATHREAD: str = "is_megathread"
    TEXT: str = "selftext"
    PROCESSED_TITLE: str = "processed_title"


@dataclass(frozen=True)
class SubRedditList:
    ANIME: str = "anime"
    BOOKS: str = "books"
    FUNNY: str = "funny"
    FOOD: str = "food"
    GAMING: str = "gaming"
    JOKES: str = "jokes"
    MEMES: str = "memes"
    MOVIES: str = "movies"
    NEWS: str = "news"
    RELATIONSHIP_ADVICE: str = "relationship_advice"
    SCIENCE: str = "science"
    SPORTS: str = "sports"
    TECHNOLOGY: str = "technology"

    @classmethod
    def get_all(cls) -> list[str]:
        """Return all available interest categories as a list"""
        return [
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


@dataclass
class RedditConstant:
    LIMIT: int = 10
    TIME_FILTER: str = "day"


@dataclass
class RedditDBConstants:
    DB_NAME: str = "reddit_posts.db"
    REDDIT_TABLE: str = "reddit_posts_table"


@dataclass
class RedditIngestorConstants:
    PRAW_SITE_NAME: str = "DEFAULT"
    PRAW_USER_AGENT: str = "Reddit Feedshift App"


class SortBy(StrEnum):
    HOT: str = "hot"
    TOP: str = "top"
    NEW: str = "new"
    RISING: str = "rising"


DEFAULT_DATABASE = "feed_shift.db"
