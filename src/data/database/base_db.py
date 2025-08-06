"""
Abstract base class for database operations across different platforms.

This module defines the common interface that all platform-specific database
handlers must implement for consistent data persistence and retrieval operations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


class BaseDB(ABC):
    """
    Abstract base class for database operations in the FeedShift application.

    This class defines the common interface that all platform-specific database
    handlers must implement. It provides a contract for data persistence operations
    including post storage, retrieval, and table management.

    All concrete database implementations must provide methods for:
    - Upserting (insert or update) post data
    - Fetching stored posts with optional filtering
    - Creating and managing database tables
    """

    @abstractmethod
    def upsert_post(self, *args: Any, **kwargs: Any) -> None:
        """
        Insert or update a post record in the database.

        This method handles both new post insertions and updates to existing posts.
        The specific implementation depends on the platform and database structure.

        Args:
            *args: Variable length argument list for platform-specific parameters
            **kwargs: Arbitrary keyword arguments that may include:
                - post data objects
                - processed content
                - metadata fields
                - timestamp information

        Raises:
            NotImplementedError: If the concrete class doesn't implement this method
            Exception: Database-specific errors during upsert operations

        Example:
            >>> db = SomeConcreteDB()
            >>> db.upsert_post(post_object, processed_title, processed_text)
        """
        logger.debug(f"upsert_post() method called on {self.__class__.__name__}")
        pass

    @abstractmethod
    def fetch_posts(self, limit: Optional[int] = None, save_df: bool = True) -> pd.DataFrame:
        """
        Retrieve posts from the database as a pandas DataFrame.

        This method fetches stored posts with optional filtering and limiting.
        It can also optionally save the retrieved data to a CSV file for caching.

        Args:
            limit (Optional[int]): Maximum number of posts to retrieve.
                If None, fetches all available posts. Defaults to None.
            save_df (bool): Whether to save the DataFrame to a CSV file.
                Useful for caching and offline analysis. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame containing post records with all relevant
                columns including content, metadata, and processed fields.
                Returns empty DataFrame if no posts are found.

        Raises:
            NotImplementedError: If the concrete class doesn't implement this method
            Exception: Database connection or query errors

        Example:
            >>> db = SomeConcreteDB()
            >>> recent_posts = db.fetch_posts(limit=100, save_df=False)
        """
        logger.debug(f"fetch_posts() method called on {self.__class__.__name__} with limit={limit}")
        pass

    @abstractmethod
    def _create_tables(self) -> None:
        """
        Create necessary database tables and indexes.

        This private method sets up the database schema required for the specific
        platform. It should be called during database initialization and should
        be idempotent (safe to call multiple times).

        The method should create all necessary tables, indexes, and constraints
        required for optimal performance and data integrity.

        Raises:
            NotImplementedError: If the concrete class doesn't implement this method
            Exception: Database schema creation errors

        Note:
            This is a private method intended for internal use during initialization.
        """
        logger.debug(f"_create_tables() method called on {self.__class__.__name__}")
        pass
