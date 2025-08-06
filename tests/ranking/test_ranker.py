"""
Simplified ranking tests.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.constants import DataCols
from src.ranking.ranker import TextRanker


class TestTextRanker:
    """Test text ranking functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for ranking tests."""
        return pd.DataFrame(
            {
                DataCols.PROCESSED_TEXT: ["text about technology", "text about sports", "text about cooking"],
                DataCols.TIMESTAMP: ["2023-01-01T10:00:00", "2023-01-01T11:00:00", "2023-01-01T12:00:00"],
            }
        )

    @patch("src.ranking.ranker.FeedShiftEmbeddor")
    @patch("src.ranking.ranker.FeedShiftDetoxified")
    def test_initialization(self, mock_detoxifier, mock_embedder, sample_data):
        """Test ranker initialization."""
        # Setup mocks
        mock_embedder_instance = Mock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.encode.return_value = np.random.rand(3, 384)

        mock_detoxifier_instance = Mock()
        mock_detoxifier.return_value = mock_detoxifier_instance

        ranker = TextRanker(sample_data)

        assert len(ranker.texts) == 3
        assert ranker.text_embeddings.shape == (3, 384)

    @patch("src.ranking.ranker.FeedShiftEmbeddor")
    @patch("src.ranking.ranker.FeedShiftDetoxified")
    def test_rerank_adds_all_scores(self, mock_detoxifier, mock_embedder, sample_data):
        """Test that reranking adds all required score columns."""
        # Setup mocks
        mock_embedder_instance = Mock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.encode.return_value = np.random.rand(3, 384)

        mock_detoxifier_instance = Mock()
        mock_detoxifier.return_value = mock_detoxifier_instance
        mock_detoxifier_instance.toxicity_score.return_value = np.array([0.1, 0.2, 0.3])

        ranker = TextRanker(sample_data)
        result = ranker.rerank(interests=["technology"])

        # Check all score columns are present
        expected_columns = [
            DataCols.UNIQUENESS_SCORE,
            DataCols.FRESHNESS_SCORE,
            DataCols.TOXICITY_SCORE,
            DataCols.INTERESTS_SCORE,
            DataCols.DIVERSITY_SCORE,
            DataCols.RECOMMENDATION_SCORE,
        ]

        for col in expected_columns:
            assert col in result.columns

    @patch("src.ranking.ranker.FeedShiftEmbeddor")
    @patch("src.ranking.ranker.FeedShiftDetoxified")
    def test_empty_interests(self, mock_detoxifier, mock_embedder, sample_data):
        """Test ranking with no interests."""
        mock_embedder_instance = Mock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.encode.return_value = np.random.rand(3, 384)

        mock_detoxifier_instance = Mock()
        mock_detoxifier.return_value = mock_detoxifier_instance

        ranker = TextRanker(sample_data)
        result = ranker._get_interests_score([])

        assert isinstance(result, np.ndarray)
        assert np.all(result == 0)  # Should be all zeros
