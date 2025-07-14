from dataclasses import dataclass


@dataclass(frozen=True)
class DataCols:
    AUTHOR = "author"
    USER = "user"
    USERNAME = "username"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    PROCESSED_TEXT = "processed_text"
    RECOMMENDATION_SCORE = "recommendation_score"
    UNIQUENESS_SCORE = "uniqueness_score"
    FRESHNESS_SCORE = "freshness_score"
    TOXICITY_SCORE = "toxicity_score"
    DIVERSITY_SCORE = "diversity_score"
    INTERESTS_SCORE = "interests_score"


@dataclass(frozen=True)
class RedditDataCols(DataCols):
    ID = "id"
    TITLE = "title"
    SCORE = "score"
    UPVOTE_RATIO = "upvote_ratio"
    NUM_COMMENTS = "num_comments"
    TIMESTAMP = "created_utc"
    SUBREDDIT = "subreddit"
    SUBSCRIBERS = "subscribers"
    NUM_CROSSPOSTS = "num_crossposts"
    POST_TYPE = "post_type"
    IS_NSFW = "is_nsfw"
    IS_BOT = "is_bot"
    IS_MEGATHREAD = "is_megathread"
    TEXT = "body"
    PROCESSED_TITLE = "processed_title"
