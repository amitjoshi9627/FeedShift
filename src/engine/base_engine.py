from pathlib import Path

import pandas as pd

from src.config.paths import RAW_DATA_PATH
from src.data.data_loader import BaseDataLoader
from src.ranking.constants import DEFAULT_TOXICITY_STRICTNESS
from src.ranking.ranker import TextRanker


class BaseEngine:
    def __init__(self, path: str | Path | None) -> None:
        if path is None:
            path = RAW_DATA_PATH
        self.data_loader = self._init_dataloader(path)
        self.text_ranker = self._init_ranker(self.data_loader.processed_data)

    @staticmethod
    def _init_dataloader(path: str):
        return BaseDataLoader(path)

    @staticmethod
    def _init_ranker(data: pd.DataFrame):
        return TextRanker(data)

    def run(
        self,
        interests: list[str],
        toxicity_strictness: float = DEFAULT_TOXICITY_STRICTNESS,
        diversity_strength: float = 0.9,
    ) -> pd.DataFrame:
        return self.text_ranker.rerank(interests, toxicity_strictness, diversity_strength=diversity_strength)
