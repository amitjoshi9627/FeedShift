"""
Simplified preprocessor tests.
"""

import pandas as pd
import pytest

from src.data.constants import RedditDataCols
from src.data.preprocessors.base_preprocessor import BasePreprocessor
from src.data.preprocessors.reddit_preprocessor import RedditPreprocessor


class TestBasePreprocessor:
    """Test base text cleaning functionality."""

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("**bold** text", "bold text"),
            ("[link](http://example.com)", "link"),
            ("https://example.com", ""),
            ("Multiple   spaces", "Multiple spaces"),
            (None, None),
            ("", ""),
        ],
    )
    def test_clean_text(self, input_text, expected):
        """Test text cleaning with various inputs."""
        result = BasePreprocessor.clean_text(input_text)
        if expected is None:
            assert result is None
        else:
            assert expected in result or result == expected


class TestRedditPreprocessor:
    """Test Reddit preprocessing."""

    @pytest.fixture
    def preprocessor(self):
        return RedditPreprocessor()

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                RedditDataCols.TITLE: ["**Bold Title**", "Normal Title"],
                RedditDataCols.TEXT: ["Text with [link](http://test.com)", "Normal text"],
                RedditDataCols.TIMESTAMP: ["2023-01-01T10:00:00", "2023-01-01T11:00:00"],
            }
        )

    def test_process_text(self, preprocessor):
        """Test individual text processing."""
        result = preprocessor.process_text("**Bold** [link](http://test.com)")
        assert "**" not in result
        assert "link" in result
        assert "http" not in result

    def test_process_data(self, preprocessor, sample_data):
        """Test full data processing."""
        result = preprocessor.process_data(sample_data)

        # Check processed columns exist
        assert RedditDataCols.PROCESSED_TITLE in result.columns
        assert RedditDataCols.PROCESSED_TEXT in result.columns

        # Check cleaning worked
        assert "**" not in result[RedditDataCols.PROCESSED_TITLE].iloc[0]
