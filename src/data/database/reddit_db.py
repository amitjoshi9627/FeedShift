"""
Reddit-specific database handler for post storage and retrieval.

This module provides SQLite-based database operations specifically designed
for Reddit post data, including metadata, content, and processed information.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from praw.reddit import Submission

from src.config.paths import DATABASE_DIR, PROCESSED_DATA_PATH
from src.data.constants import RedditDataCols, RedditDBConstants
from src.data.database.base_db import BaseDB
from src.utils.tools import save_csv

# Configure logger
logger = logging.getLogger(__name__)


class RedditDB(BaseDB):
    """
    Reddit-specific database handler using SQLite for data persistence.

    This class manages the storage and retrieval of Reddit posts with full
    metadata, processed content, and optimized querying capabilities. It provides
    efficient upsert operations to handle duplicate posts and maintains data
    integrity through proper schema design.

    Features:
    - Automatic table creation and schema management
    - Upsert operations to prevent duplicates
    - Optimized querying with sorting and limiting
    - Automatic CSV export for data analysis
    - Connection management and cleanup

    Attributes:
        db_dir (Path): Directory containing the SQLite database file
        db_path (Path): Full path to the SQLite database file
        table_name (str): Name of the Reddit posts table
        conn (sqlite3.Connection): SQLite database connection
        cursor (sqlite3.Cursor): Database cursor for query execution
    """

    def __init__(self, db_name: str = RedditDBConstants.DB_NAME) -> None:
        """
        Initialize the Reddit database handler.

        Sets up the SQLite database connection, creates necessary directories,
        and initializes the database schema for Reddit post storage.

        Args:
            db_name (str): Name of the SQLite database file.
                Defaults to RedditDBConstants.DB_NAME.

        Raises:
            RuntimeError: If database initialization or table creation fails
            PermissionError: If unable to create database directory
            sqlite3.Error: If database connection fails
        """
        logger.info(f"Initializing RedditDB with database: {db_name}")

        try:
            # Set up database paths
            self.db_dir = DATABASE_DIR
            self.db_path = self.db_dir / db_name
            self.table_name = RedditDBConstants.REDDIT_TABLE

            # Ensure directory exists
            logger.debug(f"Ensuring database directory exists: {self.db_dir}")
            self.db_dir.mkdir(parents=True, exist_ok=True)

            # Database connection with optimizations
            logger.debug(f"Connecting to database: {self.db_path}")
            self.conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)  # 30 second timeout
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()

            # Enable WAL mode for better concurrency
            self.cursor.execute("PRAGMA journal_mode=WAL")
            self.cursor.execute("PRAGMA synchronous=NORMAL")
            self.cursor.execute("PRAGMA cache_size=10000")

            # Create tables
            self._create_tables()

            logger.info(f"Successfully initialized RedditDB at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize RedditDB: {e}")
            raise

    def _create_tables(self) -> None:
        """
        Create the Reddit posts table with optimized schema and indexes.

        Creates a comprehensive table structure for Reddit posts including
        all metadata fields, processed content, and performance indexes.
        The operation is idempotent and safe to call multiple times.

        Raises:
            RuntimeError: If table creation fails
            sqlite3.Error: If SQL execution errors occur
        """
        logger.debug("Creating Reddit posts table and indexes")

        try:
            # Create main posts table
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    {RedditDataCols.ID} TEXT NOT NULL,
                    {RedditDataCols.AUTHOR} TEXT,
                    {RedditDataCols.IS_TEXT} BOOLEAN DEFAULT 0,
                    {RedditDataCols.TITLE} TEXT,
                    {RedditDataCols.SCORE} INTEGER DEFAULT 0,
                    {RedditDataCols.UPVOTE_RATIO} REAL DEFAULT 0.0,
                    {RedditDataCols.NUM_COMMENTS} INTEGER DEFAULT 0,
                    {RedditDataCols.TIMESTAMP} INTEGER NOT NULL,
                    {RedditDataCols.SUBREDDIT} TEXT,
                    {RedditDataCols.SUBSCRIBERS} INTEGER DEFAULT 0,
                    {RedditDataCols.NUM_CROSSPOSTS} INTEGER DEFAULT 0,
                    {RedditDataCols.POST_TYPE} TEXT DEFAULT 'text',
                    {RedditDataCols.IS_NSFW} BOOLEAN DEFAULT 0,
                    {RedditDataCols.IS_BOT} BOOLEAN DEFAULT 0,
                    {RedditDataCols.IS_MEGATHREAD} BOOLEAN DEFAULT 0,
                    {RedditDataCols.TEXT} TEXT,
                    {RedditDataCols.PROCESSED_TITLE} TEXT,
                    {RedditDataCols.PROCESSED_TEXT} TEXT,
                    {RedditDataCols.HASH} TEXT PRIMARY KEY,
                    last_fetched TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            self.cursor.execute(create_table_sql)

            # Create performance indexes
            indexes = [
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp "
                + f"ON {self.table_name}({RedditDataCols.TIMESTAMP} DESC)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_subreddit "
                + f"ON {self.table_name}({RedditDataCols.SUBREDDIT})",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_score ON "
                + f"{self.table_name}({RedditDataCols.SCORE} DESC)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_last_fetched ON "
                + f"{self.table_name}(last_fetched DESC)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_author ON "
                + f"{self.table_name}({RedditDataCols.AUTHOR})",
            ]

            for index_sql in indexes:
                self.cursor.execute(index_sql)

            self.conn.commit()
            logger.info(f"Successfully created table '{self.table_name}' with indexes")

        except sqlite3.Error as e:
            logger.error(f"Table creation failed: {e}")
            raise RuntimeError("Could not initialize database tables") from e

    @staticmethod
    def _get_columns_and_values(
        post: Submission, processed_title: str, processed_text: str, post_hash: str
    ) -> Dict[str, Any]:
        """
        Extract column values from a Reddit post object.

        Creates a standardized mapping between database columns and post attributes,
        handling missing attributes gracefully with sensible defaults.

        Args:
            post (Submission): PRAW Reddit submission object
            processed_title (str): Cleaned and processed post title
            processed_text (str): Cleaned and processed post content
            post_hash (str): unique hash for post

        Returns:
            Dict[str, Any]: Dictionary mapping column names to extracted values
        """
        logger.debug(f"Extracting values from Reddit post: {getattr(post, 'id', 'unknown')}")

        try:
            # Generate unique hash for deduplication

            column_values = {
                RedditDataCols.ID: str(post.id),
                RedditDataCols.AUTHOR: str(getattr(post.author, "name", "[deleted]")),
                RedditDataCols.IS_TEXT: getattr(post, "is_self", False),
                RedditDataCols.TITLE: getattr(post, "title", ""),
                RedditDataCols.SCORE: getattr(post, "score", 0),
                RedditDataCols.UPVOTE_RATIO: getattr(post, "upvote_ratio", 0.0),
                RedditDataCols.NUM_COMMENTS: getattr(post, "num_comments", 0),
                RedditDataCols.TIMESTAMP: datetime.utcfromtimestamp(
                    getattr(post, "created_utc", datetime.utcnow().timestamp())
                ).isoformat(),
                RedditDataCols.SUBREDDIT: getattr(post.subreddit, "display_name", "Unknown"),
                RedditDataCols.SUBSCRIBERS: getattr(post.subreddit, "subscribers", 0),
                RedditDataCols.NUM_CROSSPOSTS: getattr(post, "num_crossposts", 0),
                RedditDataCols.POST_TYPE: "self" if getattr(post, "is_self", False) else "link",
                RedditDataCols.IS_NSFW: getattr(post, "over_18", False),
                RedditDataCols.IS_BOT: "[bot]" in str(getattr(post.author, "name", "")).lower(),
                RedditDataCols.IS_MEGATHREAD: "megathread" in getattr(post, "title", "").lower(),
                RedditDataCols.TEXT: getattr(post, "selftext", ""),
                RedditDataCols.PROCESSED_TITLE: processed_title or "",
                RedditDataCols.PROCESSED_TEXT: processed_text or "",
                RedditDataCols.HASH: post_hash,
            }

            logger.debug(f"Successfully extracted {len(column_values)} values from post")
            return column_values

        except Exception as e:
            logger.error(f"Failed to extract values from post: {e}")
            raise

    def hash_exists(self, post_hash: str) -> bool:
        """
        Check if a post with the given hash already exists in the database.

        Args:
            post_hash (str): The hash to check for existence

        Returns:
            bool: True if hash exists, False otherwise
        """
        try:
            self.cursor.execute(
                f"SELECT 1 FROM {self.table_name} WHERE {RedditDataCols.HASH} = ? LIMIT 1", (post_hash,)
            )
            return self.cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Failed to check hash existence: {e}")
            return False  # Assume doesn't exist on error

    def upsert_post(self, post: Submission, processed_title: str, processed_text: str) -> None:
        """
        Insert or update a Reddit post in the database.

        Performs an atomic upsert operation that either inserts a new post
        or updates an existing one based on the post hash. This prevents
        duplicate posts while allowing for content updates.

        Args:
            post (Submission): PRAW Reddit submission object containing post data
            processed_title (str): Cleaned and processed version of the post title
            processed_text (str): Cleaned and processed version of the post content

        Raises:
            RuntimeError: If the database operation fails
            sqlite3.Error: If SQL execution errors occur
        """
        post_id = getattr(post, "id", "unknown")
        logger.debug(f"Upserting Reddit post: {post_id}")

        try:
            post_hash = (
                f"{post.id}_{getattr(post.author, 'name', 'deleted')}_"
                + f"{getattr(post.subreddit, 'display_name', 'Unknown')}"
            )

            if not self.hash_exists(post_hash):
                # Extract post data
                column_values = self._get_columns_and_values(post, processed_title, processed_text, post_hash)

                # Prepare SQL statement
                columns = ", ".join(column_values.keys()) + ", last_fetched"
                placeholders = ", ".join(["?"] * len(column_values)) + ", CURRENT_TIMESTAMP"
                values = list(column_values.values())

                query = f"""
                    INSERT OR REPLACE INTO {self.table_name} ({columns})
                    VALUES ({placeholders})
                """

                # Execute upsert
                self.cursor.execute(query, values)
                self.conn.commit()

                logger.debug(f"Successfully upserted post: {post_id}")
            else:
                logger.info(f"Post: {post.id} already exists")

        except sqlite3.Error as e:
            logger.error(f"Upsert failed for post {post_id}: {e}")
            self.conn.rollback()  # Rollback on error
            raise RuntimeError("Database insert or update failed") from e
        except Exception as e:
            logger.error(f"Unexpected error during upsert for post {post_id}: {e}")
            raise

    def fetch_posts(self, limit: Optional[int] = None, save_df: bool = True) -> pd.DataFrame:
        """
        Retrieve Reddit posts from the database as a pandas DataFrame.

        Fetches posts ordered by most recently fetched first, with optional
        limiting and automatic CSV export for data analysis and caching.

        Args:
            limit (Optional[int]): Maximum number of posts to retrieve.
                If None, fetches all posts. Defaults to None.
            save_df (bool): Whether to save the DataFrame to CSV file.
                Saves to PROCESSED_DATA_PATH for caching. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame containing Reddit posts with all columns.
                Returns empty DataFrame if no posts found.

        Raises:
            RuntimeError: If the database query fails
            sqlite3.Error: If SQL execution errors occur
        """
        logger.info(f"Fetching Reddit posts from database (limit: {limit})")

        try:
            # Build query with optional limit
            query = f"""
                SELECT * FROM {self.table_name}
                ORDER BY last_fetched DESC, {RedditDataCols.TIMESTAMP} DESC
            """

            if limit and limit > 0:
                query += f" LIMIT {limit}"

            # Execute query
            logger.debug(f"Executing query: {query}")
            self.cursor.execute(query)
            rows = self.cursor.fetchall()

            if not rows:
                logger.info("No posts found in database")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in rows])
            logger.info(f"Successfully retrieved {len(df)} posts from database")

            # Save to CSV if requested
            if save_df:
                try:
                    save_csv(df, PROCESSED_DATA_PATH)
                    logger.debug(f"Saved DataFrame to {PROCESSED_DATA_PATH}")
                except Exception as e:
                    logger.warning(f"Failed to save DataFrame to CSV: {e}")

            return df

        except sqlite3.Error as e:
            logger.error(f"Failed to fetch posts from database: {e}")
            raise RuntimeError("Could not fetch posts from the database") from e
        except Exception as e:
            logger.error(f"Unexpected error during post fetch: {e}")
            raise

    def get_post_count(self) -> int:
        """
        Get the total number of posts in the database.

        Returns:
            int: Total count of posts in the database

        Raises:
            RuntimeError: If the count query fails
        """
        try:
            self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            count = self.cursor.fetchone()[0]
            logger.debug(f"Database contains {count} posts")
            return count
        except sqlite3.Error as e:
            logger.error(f"Failed to get post count: {e}")
            raise RuntimeError("Could not get post count from database") from e

    def delete_old_posts(self, days_old: int = 30) -> int:
        """
        Delete posts older than specified number of days.

        Args:
            days_old (int): Number of days after which posts should be deleted

        Returns:
            int: Number of posts deleted

        Raises:
            RuntimeError: If the deletion fails
        """
        try:
            cutoff_timestamp = int((datetime.utcnow().timestamp() - (days_old * 24 * 3600)))
            query = f"DELETE FROM {self.table_name} WHERE {RedditDataCols.TIMESTAMP} < ?"

            self.cursor.execute(query, (cutoff_timestamp,))
            deleted_count = self.cursor.rowcount
            self.conn.commit()

            logger.info(f"Deleted {deleted_count} posts older than {days_old} days")
            return deleted_count

        except sqlite3.Error as e:
            logger.error(f"Failed to delete old posts: {e}")
            self.conn.rollback()
            raise RuntimeError("Could not delete old posts") from e

    def __del__(self) -> None:
        """
        Clean up database connections when object is destroyed.

        Ensures proper cleanup of database resources to prevent
        connection leaks and data corruption.
        """
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
                logger.debug("Database connection closed successfully")
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")
