import logging
import re
from abc import ABC
from typing import Optional

import pandas as pd

from src.config.paths import PROCESSED_DATA_PATH
from src.data.constants import DataCols
from src.utils.tools import save_csv

# Configure logger
logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessing across different platforms.

    This class provides common preprocessing functionality including data cleaning,
    text processing, timestamp formatting, and data persistence. Platform-specific
    preprocessors should inherit from this class and extend its functionality.

    The preprocessing pipeline includes:
    - Duplicate removal
    - Text cleaning and normalization
    - Timestamp standardization
    - Data sorting and persistence
    """

    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline for raw data.

        Performs basic data cleaning including duplicate removal and applies
        platform-specific post-processing steps. This method serves as the
        entry point for the preprocessing workflow.

        Args:
            raw_data (pd.DataFrame): Raw data DataFrame from data ingestion

        Returns:
            pd.DataFrame: Processed DataFrame with cleaned data, standardized
                timestamps, and sorted by timestamp

        Raises:
            Exception: If preprocessing operations fail

        Example:
            >>> preprocessor = SomeConcretePreprocessor()
            >>> clean_data = preprocessor.process_data(raw_df)
        """
        logger.info(f"Starting data preprocessing for {len(raw_data)} records")

        try:
            # Create deep copy and remove duplicates
            processed_data = raw_data.copy(deep=True).drop_duplicates()
            duplicates_removed = len(raw_data) - len(processed_data)

            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate records")

            logger.debug("Applying post-processing steps")
            result = self.post_processing(processed_data)

            logger.info(f"Successfully processed {len(result)} records")
            return result

        except Exception as e:
            logger.error(f"Failed to process data: {e}")
            raise

    @staticmethod
    def post_processing(data: pd.DataFrame, timestamp_col: str = DataCols.TIMESTAMP) -> pd.DataFrame:
        """
        Apply common post-processing steps to cleaned data.

        Standardizes timestamp format, sorts data chronologically, and persists
        the processed data to storage. This method is called at the end of the
        preprocessing pipeline.

        Args:
            data (pd.DataFrame): Cleaned DataFrame to post-process
            timestamp_col (str): Name of the timestamp column to format and sort by.
                Defaults to DataCols.TIMESTAMP.

        Returns:
            pd.DataFrame: Post-processed DataFrame with formatted timestamps,
                sorted chronologically, and saved to disk

        Raises:
            Exception: If timestamp conversion, sorting, or saving fails

        Example:
            >>> processed_df = BasePreprocessor.post_processing(clean_df, 'created_at')
        """
        logger.debug(f"Starting post-processing with timestamp column: {timestamp_col}")

        try:
            # Convert timestamps to standardized format
            logger.debug("Converting timestamps to standard format")
            data[timestamp_col] = pd.to_datetime(data[timestamp_col], format="mixed", dayfirst=False).dt.strftime(
                "%b %d, %Y at %H:%M"
            )

            # Sort by timestamp
            logger.debug("Sorting data by timestamp")
            data = data.sort_values(by=[timestamp_col])

            # Save processed data
            logger.info(f"Saving processed data to {PROCESSED_DATA_PATH}")
            save_csv(data, PROCESSED_DATA_PATH)

            logger.debug("Post-processing completed successfully")
            return data

        except Exception as e:
            logger.error(f"Failed during post-processing: {e}")
            raise

    @staticmethod
    def clean_text(text: str) -> Optional[str]:
        """
        Clean and normalize text content by removing formatting and URLs.

        Performs comprehensive text cleaning including:
        - Markdown link removal (preserving link text)
        - URL removal (HTTP/HTTPS and www links)
        - Markdown formatting removal (bullets, emphasis, strikethrough)
        - Whitespace normalization
        - Leading/trailing whitespace removal

        Args:
            text (str): Raw text content to clean

        Returns:
            Optional[str]: Cleaned text string, or None if input is not a string

        Example:
            >>> clean = BasePreprocessor.clean_text("[Link](http://example.com) *bold* text")
            >>> print(clean)  # "Link bold text"
        """
        if not isinstance(text, str):
            logger.debug(f"Skipping non-string input: {type(text)}")
            return None

        logger.debug("Cleaning text content")
        original_length = len(text)

        try:
            # Remove Markdown links - keep text, remove URL
            text = re.sub(r"\[(.*?)]\((.*?)\)", r"\1", text)

            # Remove bare URLs
            text = re.sub(r"https?://\S+|www\.\S+", "", text)

            # Remove Markdown formatting
            text = re.sub(r"^\s*[\*\-]\s*", "", text, flags=re.MULTILINE)  # Bullet points
            text = text.replace("*", "")  # Asterisks (bold/italic)
            text = text.replace("_", "")  # Underscores (italic)
            text = text.replace("~", "")  # Strikethrough tildes
            text = text.replace(":", ": ")  # Normalize colons

            # Normalize whitespace
            text = re.sub(r"[\n\r\t]+", " ", text)  # Replace newlines/tabs with space
            text = re.sub(r"\s+", " ", text)  # Multiple spaces to single space

            # Remove leading/trailing whitespace
            text = text.strip()

            cleaned_length = len(text)
            logger.debug(f"Text cleaned: {original_length} -> {cleaned_length} characters")

            return text

        except Exception as e:
            logger.error(f"Failed to clean text: {e}")
            return text  # Return original text if cleaning fails
