from dataclasses import dataclass


@dataclass
class RankingWeight:
    UNIQUENESS: float = 0.5
    FRESHNESS: float = 0.5

