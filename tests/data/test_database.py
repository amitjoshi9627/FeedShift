"""
Simplified database tests with local fixtures.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.data.database.reddit_db import RedditDB


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = Path(f.name)

    # Create database instance
    with pytest.MonkeyPatch().context() as m:
        m.setattr("src.data.database.reddit_db.DATABASE_DIR", temp_path.parent)
        db = RedditDB(db_name=temp_path.name)
        yield db
        db.conn.close()

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def mock_post():
    """Create a simple mock Reddit post."""
    post = Mock()
    post.id = "test123"
    post.title = "Test Title"
    post.selftext = "Test content"
    post.score = 100
    post.author.name = "testuser"
    post.subreddit.display_name = "testsubreddit"
    # Add minimal required attributes
    for attr in ["upvote_ratio", "num_comments", "created_utc", "is_self", "over_18", "num_crossposts"]:
        setattr(post, attr, 0)
    post.subreddit.subscribers = 1000
    return post


class TestRedditDB:
    """Simplified database tests."""

    def test_initialization(self, temp_db):
        """Test database initializes correctly."""
        assert temp_db.conn is not None
        assert temp_db.get_post_count() == 0

    def test_hash_exists(self, temp_db):
        """Test hash existence check."""
        assert temp_db.hash_exists("nonexistent") is False

    def test_upsert_and_fetch(self, temp_db, mock_post):
        """Test inserting and fetching posts."""
        # Insert
        temp_db.upsert_post(mock_post, "processed title", "processed text")
        assert temp_db.get_post_count() == 1

        # Fetch
        posts = temp_db.fetch_posts(save_df=False)
        assert len(posts) == 1
        assert posts.iloc[0]["id"] == "test123"

    def test_duplicate_handling(self, temp_db, mock_post):
        """Test duplicate posts are skipped."""
        temp_db.upsert_post(mock_post, "title1", "text1")
        temp_db.upsert_post(mock_post, "title2", "text2")  # Same post
        assert temp_db.get_post_count() == 1  # Should still be 1
