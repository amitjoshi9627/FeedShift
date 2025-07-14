from dataclasses import dataclass


@dataclass(frozen=True)
class RankingWeight:
    UNIQUENESS: float = 0.25
    FRESHNESS: float = 0.25
    INTERESTS: float = 0.3
    DIVERSITY: float = 0.2


DEFAULT_TOXICITY_STRICTNESS: float = 0.5
SIMILAR_POSTS_ALPHA: float = 0.25
DEFAULT_DIVERSITY_STRENGTH: float = 0.5
FRESHNESS_HALF_LIFE: int = 120
INTERESTS_SYNERGY_BOOST: float = 1.25
INTERESTS_SYNERGY_THRESHOLD: float = 0.6
