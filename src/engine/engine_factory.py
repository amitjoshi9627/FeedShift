import logging
from functools import lru_cache

from src.data.constants import SubRedditList
from src.engine import RedditEngine
from src.engine.base_engine import BaseEngine
from src.engine.constants import Platform

# Configure logger
logger = logging.getLogger(__name__)


@lru_cache(maxsize=5)
def get_engine(platform: str, subreddit: str) -> BaseEngine:
    """
    Factory function to create and return appropriate engine instances based on platform.

    This function uses LRU cache to store up to 5 engine instances for performance optimization.
    Currently, supports Reddit platform with plans to expand to other platforms.

    Args:
        platform (str): The platform identifier (e.g., Platform.REDDIT)
        subreddit (str): The subreddit name or identifier for Reddit platform

    Returns:
        BaseEngine: An instance of the appropriate engine for the specified platform

    Raises:
        ValueError: If the platform is not supported yet

    Example:
        >>> engine = get_engine(Platform.REDDIT, SubRedditList.JOKES)
        >>> results = engine.run()
    """
    logger.info(f"Creating engine for platform: {platform}, subreddit: {subreddit}")

    if platform == Platform.REDDIT:
        logger.debug(f"Initializing RedditEngine for subreddit: {subreddit}")
        return RedditEngine(subreddit)
    else:
        error_msg = f"Engine for platform '{platform}' is not implemented yet"
        logger.error(error_msg)
        raise ValueError(error_msg)


if __name__ == "__main__":
    # Configure logging for main execution
    logging.basicConfig(level=logging.INFO)

    try:
        logger.info("Starting engine factory test")
        engine = get_engine(platform=Platform.REDDIT, subreddit=SubRedditList.JOKES)
        result = engine.run().head()
        logger.info(f"Successfully retrieved {len(result)} results")
        print(result)
    except Exception as e:
        logger.error(f"Error during engine test: {e}")
        raise
