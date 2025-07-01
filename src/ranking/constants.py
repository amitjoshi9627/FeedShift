from dataclasses import dataclass


@dataclass(frozen=True)
class RankingWeight:
    UNIQUENESS: float = 0.4
    FRESHNESS: float = 0.1
    TOXICITY: float = 0.2
    INTERESTS: float = 0.1
    DIVERSITY: float = 0.2


DEFAULT_TOXICITY_STRICTNESS: float = 0.5
SIMILAR_POSTS_ALPHA: float = 0.25
DEFAULT_DIVERSITY_STRENGTH: float = 0.5
