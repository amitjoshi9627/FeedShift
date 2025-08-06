"""
Simplified engine tests.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data.constants import RedditDataCols
from src.engine.constants import Platform
from src.engine.engine_factory import get_engine
from src.engine.reddit_engine import RedditEngine


class TestEngineFactory:
    """Test engine factory."""

    @patch("src.engine.engine_factory.RedditEngine")
    def test_get_reddit_engine(self, mock_reddit_engine):
        """Test getting Reddit engine."""
        mock_instance = Mock()
        mock_reddit_engine.return_value = mock_instance

        result = get_engine(Platform.REDDIT, "test")

        assert result == mock_instance
        mock_reddit_engine.assert_called_once_with("test")

    def test_invalid_platform(self):
        """Test error for invalid platform."""
        with pytest.raises(ValueError):
            get_engine("invalid", "test")


class TestRedditEngine:
    """Test Reddit engine."""

    @patch("src.engine.reddit_engine.TextRanker")
    @patch("src.engine.reddit_engine.RedditIngestor")
    def test_initialization_and_run(self, mock_ingestor, mock_ranker):
        """Test engine initialization and basic run."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_ingestor.return_value = mock_ingestor_instance
        sample_data = pd.DataFrame(
            {RedditDataCols.TITLE: ["Post 1"], RedditDataCols.TIMESTAMP: ["2023-01-01T10:00:00"]}
        )
        mock_ingestor_instance.ingest.return_value = sample_data

        mock_ranker_instance = Mock()
        mock_ranker.return_value = mock_ranker_instance
        expected_result = pd.DataFrame({RedditDataCols.RECOMMENDATION_SCORE: [1, 2, 3]})
        mock_ranker_instance.rerank.return_value = expected_result

        # Test
        engine = RedditEngine("tech")
        result = engine.run(interests=["tech"], toxicity_strictness=0.8)

        assert result.equals(expected_result)
        mock_ranker_instance.rerank.assert_called_once_with(["tech"], 0.8, diversity_strength=0.9)
