"""
Ranking algorithm constants and configuration parameters.

This module defines weights, thresholds, and default values used by the
content ranking system to score and prioritize feed items based on
various quality and relevance metrics.
"""

import logging
from dataclasses import dataclass
from typing import Final

# Configure logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankingWeight:
    """
    Weight coefficients for different ranking factors in the scoring algorithm.

    These weights determine the relative importance of each scoring component
    in the final recommendation score. All weights should sum to 1.0 for
    proper normalization.

    Attributes:
        UNIQUENESS: Weight for content uniqueness/novelty (0.0-1.0)
        FRESHNESS: Weight for content recency/timeliness (0.0-1.0)
        INTERESTS: Weight for user interest alignment (0.0-1.0)
        DIVERSITY: Weight for content diversity promotion (0.0-1.0)

    Note:
        Current weights sum to 1.0: 0.25 + 0.25 + 0.3 + 0.2 = 1.0
    """

    UNIQUENESS: float = 0.25
    FRESHNESS: float = 0.25
    INTERESTS: float = 0.30
    DIVERSITY: float = 0.20

    def __post_init__(self) -> None:
        """Validate that weights sum to approximately 1.0."""
        total = self.UNIQUENESS + self.FRESHNESS + self.INTERESTS + self.DIVERSITY
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            logger.warning(f"Ranking weights sum to {total:.3f}, should be 1.0")


# Toxicity filtering configuration
DEFAULT_TOXICITY_STRICTNESS: Final[float] = 0.5
"""
Default strictness level for toxicity filtering (0.0-1.0).

- 0.0: No toxicity filtering applied
- 0.5: Moderate filtering (recommended default)
- 1.0: Maximum strictness, filters most potentially toxic content
"""

# Content similarity configuration
SIMILAR_POSTS_ALPHA: Final[float] = 0.25
"""
Alpha parameter for content similarity calculations.

Controls the sensitivity of duplicate/similar content detection.
Lower values make the system more sensitive to similarities.
"""

# Diversity algorithm configuration
DEFAULT_DIVERSITY_STRENGTH: Final[float] = 0.5
"""
Default strength of the diversity promotion algorithm (0.0-1.0).

- 0.0: No diversity promotion, purely score-based ranking
- 0.5: Balanced approach between score and diversity
- 1.0: Maximum diversity, potentially at the cost of relevance
"""

# Freshness scoring configuration
FRESHNESS_HALF_LIFE: Final[int] = 120
"""
Half-life in minutes for freshness score decay.

After this time period, content freshness score is reduced by 50%.
Shorter values emphasize more recent content.
"""

# Interest matching configuration
INTERESTS_SYNERGY_BOOST: Final[float] = 1.25
"""
Multiplier applied when content matches multiple user interests.

Content matching multiple interests receives this boost to its
interests score to promote well-aligned content.
"""

INTERESTS_SYNERGY_THRESHOLD: Final[float] = 0.6
"""
Threshold for applying interest synergy boost (0.0-1.0).

Content must score above this threshold in interest matching
to qualify for the synergy boost multiplier.
"""

# Score normalization constants
MIN_SCORE: Final[float] = 0.0
"""Minimum possible score for any ranking component."""

MAX_SCORE: Final[float] = 1.0
"""Maximum possible score for any ranking component."""

# Performance configuration
RANKING_BATCH_SIZE: Final[int] = 100
"""
Number of items to process in each ranking batch.

Larger batches improve throughput but use more memory.
Adjust based on available system resources.
"""

# Quality thresholds
MIN_CONTENT_LENGTH: Final[int] = 10
"""Minimum character length for content to be considered for ranking."""

MAX_CONTENT_LENGTH: Final[int] = 10000
"""Maximum character length for content processing (longer content is truncated)."""

# Caching configuration
RANKING_CACHE_TTL: Final[int] = 300
"""Time-to-live in seconds for ranking score caches."""

# Logging configuration for ranking operations
ENABLE_RANKING_DEBUG_LOGS: Final[bool] = False
"""Enable detailed debug logging for ranking operations (impacts performance)."""
