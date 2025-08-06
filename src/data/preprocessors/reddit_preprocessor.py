import logging
from typing import Optional

import pandas as pd

from src.data.constants import RedditDataCols
from src.data.preprocessors.base_preprocessor import BasePreprocessor

# Configure logger
logger = logging.getLogger(__name__)


class RedditPreprocessor(BasePreprocessor):
    """
    Reddit-specific data preprocessor for cleaning and standardizing Reddit content.

    This class extends BasePreprocessor to handle Reddit-specific data structures
    and content formatting. It processes both post titles and text content,
    applying specialized cleaning for Reddit's markdown formatting and content patterns.

    Attributes:
        Inherits all functionality from BasePreprocessor
    """

    def __init__(self) -> None:
        """
        Initialize the Reddit preprocessor.

        Sets up the preprocessor with Reddit-specific configurations
        and logging context.
        """
        super().__init__()
        logger.info("Initialized RedditPreprocessor")

    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw Reddit data through the complete preprocessing pipeline.

        Extends the base preprocessing to handle Reddit-specific columns
        including title and text content cleaning. Processes both the main
        text content and post titles separately for optimal cleaning.

        Args:
            raw_data (pd.DataFrame): Raw Reddit data containing posts with
                titles, text content, timestamps, and other metadata

        Returns:
            pd.DataFrame: Processed DataFrame with cleaned titles and text,
                standardized timestamps, and sorted chronologically

        Raises:
            KeyError: If required Reddit columns are missing
            Exception: If any preprocessing step fails

        Example:
            >>> reddit_preprocessor = RedditPreprocessor()
            >>> clean_posts = reddit_preprocessor.process_data(raw_reddit_df)
        """
        logger.info(f"Processing Reddit data with {len(raw_data)} posts")

        try:
            # Remove duplicates (inherited from base class)
            processed_data = raw_data.copy(deep=True).drop_duplicates()
            duplicates_removed = len(raw_data) - len(processed_data)

            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate Reddit posts")

            # Clean Reddit-specific text content
            logger.debug("Cleaning Reddit post text content")
            processed_data[RedditDataCols.PROCESSED_TEXT] = processed_data[RedditDataCols.TEXT].apply(self.clean_text)

            logger.debug("Cleaning Reddit post titles")
            processed_data[RedditDataCols.PROCESSED_TITLE] = processed_data[RedditDataCols.TITLE].apply(self.clean_text)

            # Log cleaning statistics
            null_text_count = processed_data[RedditDataCols.PROCESSED_TEXT].isnull().sum()
            null_title_count = processed_data[RedditDataCols.PROCESSED_TITLE].isnull().sum()

            if null_text_count > 0:
                logger.warning(f"{null_text_count} posts had text content that couldn't be processed")
            if null_title_count > 0:
                logger.warning(f"{null_title_count} posts had titles that couldn't be processed")

            # Apply post-processing with Reddit timestamp column
            logger.debug("Applying Reddit-specific post-processing")
            result = self.post_processing(processed_data, RedditDataCols.TIMESTAMP)

            logger.info(f"Successfully processed {len(result)} Reddit posts")
            return result

        except KeyError as e:
            logger.error(f"Missing required Reddit column: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to process Reddit data: {e}")
            raise

    def process_text(self, text: str) -> Optional[str]:
        """
        Process individual text content with Reddit-specific cleaning.

        Provides a convenient method for cleaning individual text strings
        using the same cleaning logic applied during bulk data processing.
        Useful for real-time text processing or testing.

        Args:
            text (str): Raw text content to clean and process

        Returns:
            Optional[str]: Cleaned text string, or None if input cannot be processed

        Raises:
            Exception: If text cleaning fails unexpectedly

        Example:
            >>> preprocessor = RedditPreprocessor()
            >>> clean = preprocessor.process_text("**Bold** text with [link](http://example.com)")
            >>> print(clean)  # "Bold text with link"
        """
        logger.debug(f"Processing individual text of length: {len(text) if isinstance(text, str) else 'N/A'}")

        try:
            result = self.clean_text(text)
            logger.debug("Individual text processing completed")
            return result

        except Exception as e:
            logger.error(f"Failed to process individual text: {e}")
            raise
