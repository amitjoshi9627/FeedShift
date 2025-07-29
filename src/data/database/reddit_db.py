import sqlite3
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from praw.reddit import Submission

from src.config.paths import DATABASE_DIR, PROCESSED_DATA_PATH
from src.data.constants import RedditDataCols, RedditDBConstants
from src.data.database import BaseDB
from src.utils.tools import save_csv

logger = logging.getLogger(__name__)


class RedditDB(BaseDB):

    def __init__(self, db_name: str = RedditDBConstants.DB_NAME):

        self.db_dir = DATABASE_DIR
        self.db_path = self.db_dir / db_name
        self.table_name = RedditDBConstants.REDDIT_TABLE

        # Ensure directory exists
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Database connection
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_tables()

        logger.info(f"Initialized database at {self.db_path}")

    def _create_tables(self) -> None:
        try:
            # User details table
            self.cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    {RedditDataCols.ID} TEXT,
                    {RedditDataCols.AUTHOR} TEXT,
                    {RedditDataCols.IS_TEXT} BOOLEAN,
                    {RedditDataCols.TITLE} TEXT,
                    {RedditDataCols.SCORE} INTEGER,
                    {RedditDataCols.UPVOTE_RATIO} REAL,
                    {RedditDataCols.NUM_COMMENTS} INTEGER,
                    {RedditDataCols.TIMESTAMP} INTEGER,
                    {RedditDataCols.SUBREDDIT} TEXT,
                    {RedditDataCols.SUBSCRIBERS} INTEGER,
                    {RedditDataCols.NUM_CROSSPOSTS} INTEGER,
                    {RedditDataCols.POST_TYPE} TEXT,
                    {RedditDataCols.IS_NSFW} BOOLEAN,
                    {RedditDataCols.IS_BOT} BOOLEAN,
                    {RedditDataCols.IS_MEGATHREAD} BOOLEAN,
                    {RedditDataCols.TEXT} TEXT,
                    {RedditDataCols.PROCESSED_TITLE} TEXT,
                    {RedditDataCols.PROCESSED_TEXT} TEXT,
                    {RedditDataCols.HASH} TEXT PRIMARY KEY,
                    last_fetched TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )

            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Table creation failed: {str(e)}")
            raise RuntimeError("Could not initialize database tables") from e

    @staticmethod
    def _get_columns_and_values(post: Submission, processed_title: str, processed_text: str):
        """
        Creates a dictionary mapping column names to values from a post object.
        This makes the value extraction logic clean and centralized.
        """

        column_values = {
            RedditDataCols.ID: str(post.id),
            RedditDataCols.AUTHOR: str(post.author),
            RedditDataCols.IS_TEXT: getattr(post, RedditDataCols.IS_TEXT, False),
            RedditDataCols.TITLE: getattr(post, RedditDataCols.TITLE, ""),
            RedditDataCols.SCORE: getattr(post, RedditDataCols.SCORE, 0),
            RedditDataCols.UPVOTE_RATIO: getattr(post, RedditDataCols.UPVOTE_RATIO, 0.0),
            RedditDataCols.NUM_COMMENTS: getattr(post, RedditDataCols.NUM_COMMENTS, 0),
            RedditDataCols.TIMESTAMP: int(getattr(post, "created_utc", datetime.utcnow().timestamp())),
            RedditDataCols.SUBREDDIT: getattr(post.subreddit, "display_name", "Unknown"),
            RedditDataCols.SUBSCRIBERS: getattr(post, RedditDataCols.SUBSCRIBERS, 0),
            RedditDataCols.NUM_CROSSPOSTS: getattr(post, RedditDataCols.NUM_CROSSPOSTS, 0),
            RedditDataCols.POST_TYPE: getattr(post, RedditDataCols.POST_TYPE, "text"),
            RedditDataCols.IS_NSFW: getattr(post, RedditDataCols.IS_NSFW, False),
            RedditDataCols.IS_BOT: getattr(post, RedditDataCols.IS_BOT, False),
            RedditDataCols.IS_MEGATHREAD: getattr(post, RedditDataCols.IS_MEGATHREAD, False),
            RedditDataCols.TEXT: getattr(post, RedditDataCols.TEXT, ""),
            RedditDataCols.PROCESSED_TITLE: processed_title,
            RedditDataCols.PROCESSED_TEXT: processed_text,
            RedditDataCols.HASH: f"{post.id}_{post.author}_{post.subreddit.display_name}",
        }

        return column_values

    def upsert_post(self, post: Submission, processed_title: str, processed_text: str):
        try:
            column_values = self._get_columns_and_values(post, processed_title, processed_text)
            columns = ", ".join(column_values.keys()) + ", last_fetched"
            placeholders = ", ".join(["?"] * len(column_values)) + ", CURRENT_TIMESTAMP"
            values = list(column_values.values())
            query = f"""
                        INSERT OR REPLACE INTO {self.table_name} ({columns})
                        VALUES ({placeholders})
                    """
            self.cursor.execute(query, values)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Upsert failed: {str(e)}")
            raise RuntimeError("Database insert or update failed") from e

    def fetch_posts(self, limit: Optional[int] = None, save_df: bool = True) -> pd.DataFrame:
        """
        Fetches all posts from the database as a pandas DataFrame.

        Args:
            limit: Optional limit on the number of rows to fetch (most recent first)
            save_df: To save df or not

        Returns:
            pd.DataFrame: DataFrame containing all reddit post records.
        """
        try:
            query = f"SELECT * FROM {self.table_name} ORDER BY last_fetched DESC"
            if limit:
                query += f" LIMIT {limit}"
            self.cursor.execute(query)
            rows = self.cursor.fetchall()

            if not rows:
                logger.info("No posts found in database.")
                return pd.DataFrame()

            df = pd.DataFrame([dict(row) for row in rows])
            if save_df:
                save_csv(df, PROCESSED_DATA_PATH)
            return df
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch posts: {str(e)}")
            raise RuntimeError("Could not fetch posts from the database") from e

    def __del__(self):
        """Clean up database connections."""
        try:
            self.conn.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.warning(f"Error closing connection: {str(e)}")
