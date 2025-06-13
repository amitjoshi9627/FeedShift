import os.path
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.models.constants import (
    EMBEDDING_MODEL_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
    MODEL_DIR,
)


class FeedShiftEmbeddor:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        model_path: str | Path = EMBEDDING_MODEL_PATH,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.device = self._detect_device()
        self.model = self.load_model()

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self):
        if self.model_path.exists():
            embedding_model = SentenceTransformer(str(self.model_path))
        else:
            embedding_model = SentenceTransformer(self.model_name)
            self._save_model(embedding_model)

        return embedding_model.to(self.device)

    def _save_model(self, model):
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(str(self.model_path))

    def encode(
        self, sentences: str | list[str], batch_size: int = EMBEDDING_MODEL_BATCH_SIZE
    ) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        with torch.inference_mode():
            embeddings = self.model.encode(
                sentences, batch_size=batch_size, device=self.device
            )
        return embeddings

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


if __name__ == "__main__":
    emb = FeedShiftEmbeddor()
    sentence = ["AJ is good", "Nope, not"]

    print(emb(sentence).shape)
