import re

import numpy as np
from detoxify import Detoxify

from src.models.constants import DEFAULT_NON_TOXIC_SCORE, DEFAULT_TOXIC_SCORE


class DetoxifyModelError(Exception):
    """Raised when Detoxify model fails to load or predict."""


class FeedShiftDetoxified:
    def __init__(self):
        try:
            self.model = Detoxify("original")
        except DetoxifyModelError:
            self.model = None

        self.toxic_words = [
            r"\b(fuck|shit|bitch|asshole|bastard|dumb|retard|suck|hate|idiot|kill|die)\b",
            r"\b(stupid|moron|jerk|ugly|loser|trash|racist|sexist|terrorist)\b",
        ]
        self.default_toxic_score = DEFAULT_TOXIC_SCORE
        self.default_non_toxic_score = DEFAULT_NON_TOXIC_SCORE

    def _is_toxic_regex(self, text: str) -> bool:
        is_toxic = any(re.search(pattern, text.lower()) for pattern in self.toxic_words)
        return self.default_toxic_score if is_toxic else self.default_non_toxic_score

    def toxicity_score(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.model:
            try:
                score = np.max(
                    np.column_stack(list(self.model.predict(texts).values())), axis=1
                ).reshape(-1, 1)
            except DetoxifyModelError:
                score = np.array([self._is_toxic_regex(text) for text in texts])
        else:
            score = np.array([self._is_toxic_regex(text) for text in texts])

        return score
