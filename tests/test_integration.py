"""
Simplified end-to-end integration test.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.constants import DataCols
from src.engine.constants import Platform
from src.engine.engine_factory import get_engine


def create_mock_posts(count=5):
    """Create mock Reddit posts."""
    posts = []
    for i in range(count):
        post = Mock()
        post.id = f"post_{i}"
        post.title = f"Test Post {i}"
        post.selftext = f"Content about technology and science {i}"
        post.author.name = f"user_{i}"
        post.subreddit.display_name = "technology"
        post.subreddit.subscribers = 1000000
        # Add required attributes with defaults
        for attr in ["score", "upvote_ratio", "num_comments", "is_self", "over_18", "num_crossposts"]:
            setattr(post, attr, i)
        post.created_utc = time.time() - (i * 3600)
        posts.append(post)
    return posts


@patch("src.data.ingestors.reddit_ingestor.praw.Reddit")
@patch("src.models.embedder.SentenceTransformer")
@patch("src.models.detoxifier.Detoxify")
@patch("src.models.embedder.torch")
def test_full_pipeline(mock_torch, mock_detoxify, mock_st, mock_praw):
    """Test complete pipeline from ingestion to ranking."""

    # Setup all mocks
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False

    mock_reddit = Mock()
    mock_praw.return_value = mock_reddit
    mock_subreddit = Mock()
    mock_reddit.subreddit.return_value = mock_subreddit
    mock_subreddit.top.return_value = create_mock_posts(3)

    mock_model = Mock()
    mock_st.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4]] * 3

    mock_detoxify_instance = Mock()
    mock_detoxify.return_value = mock_detoxify_instance
    mock_detoxify_instance.toxicity_score.return_value = [0.1, 0.2, 0.3]

    # Test with temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("src.data.database.reddit_db.DATABASE_DIR", Path(temp_dir)):
            # Run complete pipeline
            engine = get_engine(Platform.REDDIT, "technology")
            result = engine.run(interests=["technology", "science"], toxicity_strictness=0.5)

            # Verify results
            assert len(result) == 3
            assert DataCols.RECOMMENDATION_SCORE in result.columns

            # Check scores are sorted
            scores = result[DataCols.RECOMMENDATION_SCORE].values
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
