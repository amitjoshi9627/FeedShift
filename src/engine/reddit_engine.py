import logging
from abc import ABC
from typing import List, Optional

import pandas as pd

from src.data.constants import RedditDataCols
from src.data.ingestors.reddit_ingestor import RedditIngestor
from src.engine.base_engine import BaseEngine
from src.ranking.constants import DEFAULT_TOXICITY_STRICTNESS
from src.ranking.ranker import TextRanker

# Configure logger
logger = logging.getLogger(__name__)


class RedditEngine(BaseEngine, ABC):
    """
    Reddit-specific implementation of the BaseEngine for processing Reddit feeds.

    This engine handles the ingestion of Reddit data from specified subreddits,
    applies text ranking algorithms, and provides reranked results based on
    user interests, toxicity filtering, and diversity parameters.

    Attributes:
        ingestor (RedditIngestor): Handler for Reddit data ingestion
        text_ranker (TextRanker): Ranking algorithm for text-based content
    """

    def __init__(self, subreddit: str) -> None:
        """
        Initialize the Reddit engine with a specific subreddit.

        Args:
            subreddit (str): The name/identifier of the subreddit to process

        Raises:
            Exception: If data ingestion or text ranker initialization fails
        """
        logger.info(f"Initializing RedditEngine for subreddit: {subreddit}")

        try:
            self.ingestor = RedditIngestor()
            logger.debug("RedditIngestor initialized successfully")

            # Ingest data from the specified subreddit
            logger.info(f"Starting data ingestion for subreddit: {subreddit}")
            data = self.ingestor.ingest(subreddit)
            logger.info(f"Successfully ingested {len(data)} posts from r/{subreddit}")

            # Initialize text ranker with ingested data
            self.text_ranker = TextRanker(data, timestamp_col=RedditDataCols.TIMESTAMP, text_col=RedditDataCols.TITLE)
            logger.debug("TextRanker initialized with Reddit data")

        except Exception as e:
            logger.error(f"Failed to initialize RedditEngine for subreddit '{subreddit}': {e}")
            raise

    def run(
        self,
        interests: Optional[List[str]] = None,
        toxicity_strictness: float = DEFAULT_TOXICITY_STRICTNESS,
        diversity_strength: float = 0.9,
    ) -> pd.DataFrame:
        """
        Execute the Reddit feed processing and ranking pipeline.

        This method applies the configured ranking algorithm to reorder Reddit posts
        based on user interests, toxicity filtering, and diversity requirements.

        Args:
            interests (Optional[List[str]]): List of user interest keywords/topics.
                If None or empty, no interest-based filtering is applied.
            toxicity_strictness (float): Strictness level for toxicity filtering
                (0.0 = no filtering, 1.0 = maximum filtering).
                Defaults to DEFAULT_TOXICITY_STRICTNESS.
            diversity_strength (float): Strength of diversity algorithm
                (0.0 = no diversity, 1.0 = maximum diversity).
                Defaults to 0.9.

        Returns:
            pd.DataFrame: Reranked DataFrame containing Reddit posts ordered by
                relevance, toxicity score, and diversity metrics.

        Raises:
            Exception: If the ranking process fails
        """
        interests = interests if interests is not None else []

        logger.info("Running Reddit feed processing with parameters:")
        logger.info(f"  - Interests: {interests}")
        logger.info(f"  - Toxicity strictness: {toxicity_strictness}")
        logger.info(f"  - Diversity strength: {diversity_strength}")

        try:
            result = self.text_ranker.rerank(interests, toxicity_strictness, diversity_strength=diversity_strength)

            logger.info(f"Successfully processed and reranked {len(result)} Reddit posts")

            return result

        except Exception as e:
            logger.error(f"Failed to process Reddit feed: {e}")
            raise
