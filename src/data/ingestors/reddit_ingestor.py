import logging
import os
from typing import Iterator

import pandas as pd
import praw
from dotenv import load_dotenv
from praw.models import Submission, Subreddit

from src.config.paths import ENV_FILE_PATH
from src.data.constants import RedditConstant, RedditDataCols, RedditIngestorConstants, SortBy, SubRedditList
from src.data.database.reddit_db import RedditDB
from src.data.ingestors.base_ingestor import BaseIngestor
from src.data.preprocessors.reddit_preprocessor import RedditPreprocessor

# Configure logger
logger = logging.getLogger(__name__)


class RedditIngestor(BaseIngestor):
    """
    Reddit-specific implementation of BaseIngestor for fetching Reddit posts.

    This class handles the complete Reddit data ingestion pipeline including:
    - Authentication with Reddit API using PRAW
    - Fetching posts from specified subreddits with various sorting options
    - Preprocessing and cleaning post content
    - Database storage and retrieval operations

    Supports multiple sorting methods (top, new, hot, rising) and configurable
    limits and time filters for flexible data collection.

    Attributes:
        reddit_praw (praw.Reddit): Authenticated Reddit API client
        reddit_db (RedditDB): Database handler for Reddit data operations
        preprocessor (RedditPreprocessor): Text preprocessing handler
    """

    def __init__(self) -> None:
        """
        Initialize the Reddit ingestor with API credentials and dependencies.

        Sets up PRAW Reddit client using environment variables, initializes
        database connection, and configures the text preprocessor for Reddit content.

        Raises:
            Exception: If environment variables are missing, API authentication fails,
                      or database connection cannot be established

        Environment Variables Required:
            - REDDIT_CLIENT_ID: Reddit API client ID
            - REDDIT_CLIENT_SECRET: Reddit API client secret
        """
        super().__init__()
        logger.info("Initializing RedditIngestor")

        try:
            # Load environment variables
            load_dotenv(dotenv_path=ENV_FILE_PATH)
            logger.debug(f"Loaded environment variables from {ENV_FILE_PATH}")

            # Initialize Reddit API client
            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")

            if not client_id or not client_secret:
                raise ValueError("Reddit API credentials not found in environment variables")

            self.reddit_praw = praw.Reddit(
                RedditIngestorConstants.PRAW_SITE_NAME,
                client_id=client_id,
                user_agent=RedditIngestorConstants.PRAW_USER_AGENT,
                client_secret=client_secret,
            )
            logger.info("Successfully initialized Reddit PRAW client")

            # Initialize database and preprocessor
            self.reddit_db = RedditDB()
            self.preprocessor = RedditPreprocessor()
            logger.debug("Initialized database handler and preprocessor")

        except Exception as e:
            logger.error(f"Failed to initialize RedditIngestor: {e}")
            raise

    def ingest(
        self,
        subreddit: str,
        limit: int = RedditConstant.LIMIT,
        time_filter: str = RedditConstant.TIME_FILTER,
        sort_by: str = SortBy.TOP,
    ) -> pd.DataFrame:
        """
        Ingest Reddit posts from a specified subreddit with configurable parameters.

        Fetches posts using the specified sorting method, processes the content,
        stores it in the database, and returns the ingested data.

        Args:
            subreddit (str): Name of the subreddit to fetch posts from (without 'r/')
            limit (int): Maximum number of posts to fetch. Defaults to RedditConstant.LIMIT
            time_filter (str): Time filter for 'top' sorting ('hour', 'day', 'week',
                              'month', 'year', 'all'). Defaults to RedditConstant.TIME_FILTER
            sort_by (str): Sorting method for posts. Options: 'top', 'new', 'hot', 'rising'.
                          Defaults to SortBy.TOP

        Returns:
            pd.DataFrame: DataFrame containing all ingested and processed Reddit posts

        Raises:
            ValueError: If an unsupported sort_by method is provided
            Exception: If subreddit access fails, API limits are exceeded, or database operations fail

        Example:
            >>> ingestor = RedditIngestor()
            >>> posts = ingestor.ingest('jokes', limit=50, sort_by='hot')
        """
        logger.info(f"Starting ingestion from r/{subreddit}")
        logger.info(f"Parameters: limit={limit}, time_filter={time_filter}, sort_by={sort_by}")

        try:
            # Get subreddit object
            subreddit_obj = self.reddit_praw.subreddit(subreddit)
            logger.debug(f"Successfully accessed subreddit: r/{subreddit}")

            # Fetch posts based on sorting method
            if sort_by == SortBy.TOP:
                posts = self._fetch_top_posts(subreddit_obj, limit, time_filter)
            elif sort_by == SortBy.NEW:
                posts = self._fetch_new_posts(subreddit_obj, limit)
            elif sort_by == SortBy.HOT:
                posts = self._fetch_hot_posts(subreddit_obj, limit)
            elif sort_by == SortBy.RISING:
                posts = self._fetch_rising_posts(subreddit_obj, limit)
            else:
                error_msg = f"Unsupported sorting type: {sort_by}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Successfully fetched posts using {sort_by} sorting")

            # Insert posts into database
            logger.debug("Starting post insertion into database")
            self._insert_posts(posts)

            # Fetch and return all posts
            result = self.reddit_db.fetch_posts()
            logger.info(f"Ingestion completed successfully. Total posts in database: {len(result)}")

            return result

        except Exception as e:
            logger.error(f"Failed to ingest posts from r/{subreddit}: {e}")
            raise

    def _insert_posts(self, posts: Iterator[Submission]) -> None:
        """
        Process and insert Reddit posts into the database.

        Iterates through the fetched posts, applies text preprocessing to titles
        and content, and performs database upsert operations to avoid duplicates.

        Args:
            posts (Iterator[Submission]): Iterator of Reddit submission objects

        Raises:
            Exception: If post processing or database insertion fails
        """
        logger.debug("Processing and inserting posts into database")

        try:
            processed_count = 0
            for post in posts:
                # Process post content
                processed_title = self.preprocessor.process_text(getattr(post, RedditDataCols.TITLE, ""))
                processed_text = self.preprocessor.process_text(getattr(post, RedditDataCols.TEXT, ""))

                # Insert into database
                self.reddit_db.upsert_post(post, processed_title, processed_text)
                processed_count += 1

                if processed_count % 10 == 0:  # Log progress every 10 posts
                    logger.debug(f"Processed {processed_count} posts")

            logger.info(f"Successfully processed and inserted {processed_count} posts")

        except Exception as e:
            logger.error(f"Failed to insert posts: {e}")
            raise

    def load_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Load previously ingested Reddit posts from the database.

        Retrieves stored Reddit posts up to the specified limit for use by
        downstream processing components like ranking engines.

        Args:
            limit (int): Maximum number of posts to retrieve. Defaults to 100

        Returns:
            pd.DataFrame: DataFrame containing Reddit posts with columns for
                titles, text content, timestamps, and processed versions

        Raises:
            Exception: If database query fails

        Example:
            >>> ingestor = RedditIngestor()
            >>> recent_posts = ingestor.load_data(limit=50)
        """
        logger.info(f"Loading Reddit data with limit={limit}")

        try:
            result = self.reddit_db.fetch_posts(limit=limit)
            logger.info(f"Successfully loaded {len(result)} Reddit posts from database")
            return result

        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            raise

    @staticmethod
    def _fetch_top_posts(subreddit: Subreddit, limit: int, time_filter: str) -> Iterator[Submission]:
        """
        Fetch top posts from a subreddit with time filtering.

        Args:
            subreddit (Subreddit): PRAW Subreddit object
            limit (int): Maximum number of posts to fetch
            time_filter (str): Time period for top posts ('hour', 'day', 'week', etc.)

        Returns:
            Iterator[Submission]: Iterator of Reddit submission objects
        """
        logger.debug(f"Fetching top {limit} posts with time_filter={time_filter}")
        return subreddit.top(limit=limit, time_filter=time_filter)

    @staticmethod
    def _fetch_new_posts(subreddit: Subreddit, limit: int) -> Iterator[Submission]:
        """
        Fetch the newest posts from a subreddit.

        Args:
            subreddit (Subreddit): PRAW Subreddit object
            limit (int): Maximum number of posts to fetch

        Returns:
            Iterator[Submission]: Iterator of Reddit submission objects
        """
        logger.debug(f"Fetching {limit} new posts")
        return subreddit.new(limit=limit)

    @staticmethod
    def _fetch_hot_posts(subreddit: Subreddit, limit: int) -> Iterator[Submission]:
        """
        Fetch hot posts from a subreddit.

        Args:
            subreddit (Subreddit): PRAW Subreddit object
            limit (int): Maximum number of posts to fetch

        Returns:
            Iterator[Submission]: Iterator of Reddit submission objects
        """
        logger.debug(f"Fetching {limit} hot posts")
        return subreddit.hot(limit=limit)

    @staticmethod
    def _fetch_rising_posts(subreddit: Subreddit, limit: int) -> Iterator[Submission]:
        """
        Fetch rising posts from a subreddit.

        Args:
            subreddit (Subreddit): PRAW Subreddit object
            limit (int): Maximum number of posts to fetch

        Returns:
            Iterator[Submission]: Iterator of Reddit submission objects
        """
        logger.debug(f"Fetching {limit} rising posts")
        return subreddit.rising(limit=limit)


if __name__ == "__main__":
    # Configure logging for main execution
    logging.basicConfig(level=logging.INFO)

    try:
        logger.info("Starting Reddit ingestor test")
        ingestor = RedditIngestor()

        # Ingest data from jokes subreddit
        ingestor.ingest(subreddit=SubRedditList.JOKES)

        # Load and display processed text
        reddit_posts = ingestor.load_data()
        logger.info(f"Loaded {len(reddit_posts)} posts for display")
        print(reddit_posts["processed_text"])

    except Exception as e:
        logger.error(f"Error during Reddit ingestor test: {e}")
        raise
