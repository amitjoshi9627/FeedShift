from dataclasses import dataclass


@dataclass
class RankingWeight:
    UNIQUENESS: float = 0.125
    FRESHNESS: float = 0.125
    TOXICITY: float = 0.25
    INTERESTS: float = 0.5


DEFAULT_TOXICITY_STRICTNESS = 0.5
