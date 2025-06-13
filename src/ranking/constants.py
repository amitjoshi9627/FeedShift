from dataclasses import dataclass


@dataclass
class RankingWeight:
    UNIQUENESS: float = 0.25
    FRESHNESS: float = 0.25
    TOXICITY: float = 0.5


DEFAULT_TOXICITY_STRICTNESS = 0.5
