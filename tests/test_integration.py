from unittest.mock import Mock, patch

from src.engine.constants import Platform
from src.engine.engine_factory import get_engine


@patch("src.data.ingestors.reddit_ingestor.praw.Reddit")
@patch("src.models.embedder.SentenceTransformer")
@patch("src.models.detoxifier.Detoxify")
def test_full_pipeline(mock_detoxify, mock_st, mock_praw):
    """Test complete pipeline from ingestion to ranking."""

    # Mock Reddit API
    mock_reddit = Mock()
    mock_praw.return_value = mock_reddit
    mock_subreddit = Mock()
    mock_reddit.subreddit.return_value = mock_subreddit

    # Create mock posts
    mock_posts = []
    for i in range(3):
        post = Mock()
        post.id = f"post_{i}"
        post.title = f"Test Post {i}"
        post.selftext = f"Content about technology {i}"
        post.author.name = f"user_{i}"
        post.subreddit.display_name = "technology"
        post.subreddit.subscribers = 1000000
        # Add required attributes
        for attr in ["score", "upvote_ratio", "num_comments", "is_self", "over_18", "num_crossposts"]:
            setattr(post, attr, i)
        post.created_utc = 1234567890 - (i * 3600)
        mock_posts.append(post)

    mock_subreddit.top.return_value = mock_posts

    # Mock ML models
    mock_model = Mock()
    mock_st.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4]] * 3

    mock_detoxify_instance = Mock()
    mock_detoxify.return_value = mock_detoxify_instance
    mock_detoxify_instance.toxicity_score.return_value = [[0.1], [0.2], [0.3]]

    # Test the pipeline
    engine = get_engine(Platform.REDDIT, "technology")
    result = engine.run(interests=["technology"], toxicity_strictness=0.5)

    # Verify results
    assert len(result) == 3
    assert all(col in result.columns for col in ["id", "title", "recommendation_score"])

    # Check that scores are sorted descending
    scores = result["recommendation_score"].values
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
