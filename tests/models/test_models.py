"""
Simplified model tests.
"""

from unittest.mock import Mock, patch

import numpy as np

from src.models.detoxifier import FeedShiftDetoxified
from src.models.embedder import FeedShiftEmbeddor


class TestFeedShiftEmbeddor:
    """Test text embedding functionality."""

    @patch("src.models.embedder.SentenceTransformer")
    @patch("src.models.embedder.torch")
    def test_initialization_and_encoding(self, mock_torch, mock_st):
        """Test embedder initialization and basic encoding."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        mock_model = Mock()
        mock_st.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        # Test
        embedder = FeedShiftEmbeddor()
        result = embedder.encode("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)

    @patch("src.models.embedder.SentenceTransformer")
    @patch("src.models.embedder.torch")
    def test_empty_input(self, mock_torch, mock_st):
        """Test handling of empty input."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_model = Mock()
        mock_st.return_value = mock_model
        mock_model.to.return_value = mock_model

        embedder = FeedShiftEmbeddor()
        result = embedder.encode([])

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 0


class TestFeedShiftDetoxified:
    """Test toxicity detection functionality."""

    @patch("src.models.detoxifier.Detoxify")
    def test_with_working_model(self, mock_detoxify):
        """Test toxicity detection with working ML model."""
        mock_model = Mock()
        mock_detoxify.return_value = mock_model
        mock_model.predict.return_value = {
            "toxicity": [0.8],
            "severe_toxicity": [0.1],
            "obscene": [0.3],
            "threat": [0.1],
            "insult": [0.2],
            "identity_attack": [0.1],
        }

        detoxifier = FeedShiftDetoxified()
        result = detoxifier.toxicity_score("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.8  # Max score

    @patch("src.models.detoxifier.Detoxify")
    def test_fallback_to_regex(self, mock_detoxify):
        """Test fallback to regex when model fails."""
        mock_detoxify.side_effect = Exception("Model failed")

        detoxifier = FeedShiftDetoxified()

        # Test toxic content
        result_toxic = detoxifier.toxicity_score("this is shit")
        assert result_toxic[0, 0] == detoxifier.default_toxic_score

        # Test clean content
        result_clean = detoxifier.toxicity_score("this is nice content")
        assert result_clean[0, 0] == detoxifier.default_non_toxic_score
