import os
from typing import Iterator

import pandas as pd
import praw
from dotenv import load_dotenv
from praw.models import Subreddit

from src.config.paths import ENV_FILE_PATH
from src.data.database.reddit_db import RedditDB
from src.data.ingestors import BaseIngestor
from src.data.constants import SubRedditList, RedditConstant, SortBy, RedditDataCols, RedditIngestorConstants
from src.data.preprocessors import RedditPreprocessor


class RedditIngestor(BaseIngestor):
    def __init__(self):
        super().__init__()
        load_dotenv(dotenv_path=ENV_FILE_PATH)
        self.reddit_praw = praw.Reddit(
            RedditIngestorConstants.PRAW_SITE_NAME,
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            user_agent=RedditIngestorConstants.PRAW_USER_AGENT,
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        )
        self.reddit_db = RedditDB()
        self.preprocessor = RedditPreprocessor()

    def ingest(
        self,
        subreddit: str,
        limit=RedditConstant.LIMIT,
        time_filter=RedditConstant.TIME_FILTER,
        sort_by: str = SortBy.TOP,
    ):
        subreddit = self.reddit_praw.subreddit(subreddit)
        if sort_by == SortBy.TOP:
            posts = self._fetch_top_posts(subreddit, limit, time_filter)
        elif sort_by == SortBy.NEW:
            posts = self._fetch_new_posts(subreddit, limit)
        elif sort_by == SortBy.HOT:
            posts = self._fetch_hot_posts(subreddit, limit)
        elif sort_by == SortBy.RISING:
            posts = self._fetch_rising_posts(subreddit, limit)
        else:
            raise ValueError(f"Posts aren't available by Sorting type: {sort_by}")

        self._insert_posts(posts)
        return self.reddit_db.fetch_posts()

    def _insert_posts(self, posts):
        for post in posts:
            processed_title = self.preprocessor.process_text(getattr(post, RedditDataCols.TITLE, ""))
            processed_text = self.preprocessor.process_text(getattr(post, RedditDataCols.TEXT, ""))
            self.reddit_db.upsert_post(post, processed_title, processed_text)

    def load_data(self, limit: int = 100) -> pd.DataFrame:
        return self.reddit_db.fetch_posts(limit=limit)

    @staticmethod
    def _fetch_top_posts(subreddit: Subreddit, limit: int, time_filter: str) -> Iterator:
        return subreddit.top(limit=limit, time_filter=time_filter)

    @staticmethod
    def _fetch_new_posts(subreddit: Subreddit, limit: int) -> Iterator:
        return subreddit.new(limit=limit)

    @staticmethod
    def _fetch_hot_posts(subreddit: Subreddit, limit: int) -> Iterator:
        return subreddit.hot(limit=limit)

    @staticmethod
    def _fetch_rising_posts(subreddit: Subreddit, limit: int) -> Iterator:
        return subreddit.rising(limit=limit)


if __name__ == "__main__":
    ingestor = RedditIngestor()
    ingestor.ingest(subreddit=SubRedditList.JOKES)
    reddit_posts = ingestor.load_data()
    print(reddit_posts["processed_text"])
