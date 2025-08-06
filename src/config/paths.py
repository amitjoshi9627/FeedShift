"""
Path configuration and directory structure for the FeedShift application.

This module defines all file paths, directory locations, and path-related
utilities used throughout the application for data storage, configuration,
and model management.
"""

import logging
from pathlib import Path
from typing import Final

# Configure logger
logger = logging.getLogger(__name__)

# Base directory configuration
ROOT_DIR: Final[Path] = Path(__file__).parents[2]
"""Root directory of the FeedShift application."""

CONFIG_DIR: Final[Path] = Path(__file__).parent
"""Configuration files directory."""

# Data storage paths
DATA_DIR: Final[Path] = ROOT_DIR / ".data_cache"
"""Main directory for all data storage and caching."""

RAW_DATA_PATH: Final[Path] = DATA_DIR / "raw" / "raw_data.csv"
"""Path for storing raw, unprocessed data files."""

PROCESSED_DATA_PATH: Final[Path] = DATA_DIR / "processed" / "reddit_processed_data.csv"
"""Path for storing cleaned and processed data files."""

DATABASE_DIR: Final[Path] = DATA_DIR / "database"
"""Directory for SQLite database files."""

# Configuration file paths
PRAW_INI_PATH: Final[Path] = CONFIG_DIR / "praw.ini"
"""Path to PRAW (Reddit API) configuration file."""

ENV_FILE_PATH: Final[Path] = CONFIG_DIR / ".env"
"""Path to environment variables file containing API keys and secrets."""

# Logging configuration
LOGS_DIR: Final[Path] = ROOT_DIR / "logs"
"""Directory for application log files."""

LOG_FILE_PATH: Final[Path] = LOGS_DIR / "feedshift.log"
"""Main application log file path."""

# Backup and archive paths
BACKUP_DIR: Final[Path] = DATA_DIR / "backups"
"""Directory for data backups and archives."""

TEMP_DIR: Final[Path] = DATA_DIR / "temp"
"""Directory for temporary files and processing."""


def ensure_directories() -> None:
    """
    Create all necessary directories if they don't exist.

    This function should be called during application initialization
    to ensure all required directories are available.

    Raises:
        PermissionError: If unable to create directories due to permissions
        OSError: If directory creation fails for other reasons
    """
    directories = [
        DATA_DIR,
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        DATABASE_DIR,
        LOGS_DIR,
        BACKUP_DIR,
        TEMP_DIR,
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise


def get_database_path(db_name: str) -> Path:
    """
    Get full path for a database file.

    Args:
        db_name (str): Database filename

    Returns:
        Path: Full path to the database file
    """
    return DATABASE_DIR / db_name


def get_backup_path(filename: str) -> Path:
    """
    Get full path for a backup file with timestamp.

    Args:
        filename (str): Original filename to back up

    Returns:
        Path: Full path to the backup file with timestamp
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
    return BACKUP_DIR / backup_filename
