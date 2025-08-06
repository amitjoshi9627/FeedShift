"""
Text embedding module for semantic similarity and content analysis.

This module provides text embedding capabilities using SentenceTransformers
for semantic similarity calculations, content clustering, and interest matching
in the FeedShift recommendation system.
"""

import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.models.constants import (
    EMBEDDING_MODEL_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_VECTOR_SIZE,
    MODEL_DIR,
)

# Configure logger
logger = logging.getLogger(__name__)


class FeedShiftEmbeddor:
    """
    Advanced text embedding system for semantic similarity and content analysis.

    This class provides high-quality text embeddings using SentenceTransformers
    with automatic model caching, device optimization, and batch processing
    capabilities. It supports both local model storage and automatic downloading
    from HuggingFace.

    Features:
    - Automatic device detection (CUDA, MPS, CPU)
    - Local model caching for offline usage
    - Batch processing for efficiency
    - Optimized inference with torch.inference_mode()
    - Flexible input handling (string or list)
    - Comprehensive error handling and logging

    Attributes:
        model_name (str): HuggingFace model identifier
        model_path (Path): Local path for model storage
        device (str): Computing device (cuda/mps/cpu)
        model (SentenceTransformer): Loaded embedding model
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        model_path: Union[str, Path] = EMBEDDING_MODEL_PATH,
    ) -> None:
        """
        Initialize the text embedding system.

        Sets up the embedding model with automatic device detection and
        local caching capabilities. Downloads the model if not available locally.

        Args:
            model_name (str): HuggingFace model identifier for downloading.
                Defaults to EMBEDDING_MODEL_NAME constant.
            model_path (Union[str, Path]): Local path for model storage.
                Defaults to EMBEDDING_MODEL_PATH constant.

        Raises:
            Exception: If model loading or initialization fails

        """
        logger.info("Initializing FeedShiftEmbeddor")

        try:
            self.model_name = model_name
            self.model_path = Path(model_path)

            # Auto-detect optimal device
            self.device = self._detect_device()
            logger.info(f"Using device: {self.device}")

            # Load or download model
            self.model = self.load_model()

            logger.info(f"Successfully initialized embedder with model: {model_name}")
            logger.debug(f"Model path: {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize FeedShiftEmbeddor: {e}")
            raise

    @staticmethod
    def _detect_device() -> str:
        """
        Auto-detect the best available computing device.

        Checks for CUDA GPU, Apple Metal Performance Shaders (MPS), or falls
        back to CPU. Prioritizes devices based on performance capabilities.

        Returns:
            str: Device identifier ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available: {device_name}")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Apple MPS available")
            return "mps"
        else:
            logger.info("Using CPU for inference")
            return "cpu"

    def load_model(self) -> SentenceTransformer:
        """
        Load the SentenceTransformer model from local cache or download.

        Attempts to load from local cache first for faster initialization.
        If not available locally, downloads from HuggingFace and caches
        for future use.

        Returns:
            SentenceTransformer: Loaded and configured embedding model

        Raises:
            Exception: If model loading or downloading fails
        """
        logger.debug(f"Loading embedding model from: {self.model_path}")

        try:
            # Try loading from local cache first
            if self.model_path.exists() and any(self.model_path.iterdir()):
                logger.info(f"Loading cached model from: {self.model_path}")
                embedding_model = SentenceTransformer(str(self.model_path))
                logger.info("Successfully loaded cached model")
            else:
                # Download and cache model
                logger.info(f"Downloading model: {self.model_name}")
                embedding_model = SentenceTransformer(self.model_name)

                # Save for future use
                logger.info("Caching model for future use")
                self._save_model(embedding_model)
                logger.info(f"Model cached to: {self.model_path}")

            # Move model to optimal device
            logger.debug(f"Moving model to device: {self.device}")
            embedding_model = embedding_model.to(self.device)

            # Log model information
            if hasattr(embedding_model, "get_sentence_embedding_dimension"):
                dim = embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Model embedding dimension: {dim}")

            return embedding_model

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _save_model(self, model: SentenceTransformer) -> None:
        """
        Save the model to local cache directory.

        Creates necessary directories and saves the model for offline usage
        and faster future initialization.

        Args:
            model (SentenceTransformer): Model to save

        Raises:
            Exception: If model saving fails
        """
        try:
            # Ensure directory exists
            logger.debug(f"Creating model directory: {MODEL_DIR}")
            os.makedirs(MODEL_DIR, exist_ok=True)

            # Save model
            logger.debug(f"Saving model to: {self.model_path}")
            model.save(str(self.model_path))

            # Verify save was successful
            if self.model_path.exists():
                logger.info("Model saved successfully")
            else:
                logger.error("Model save verification failed")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def encode(self, sentences: Union[str, List[str]], batch_size: int = EMBEDDING_MODEL_BATCH_SIZE) -> np.ndarray:
        """
        Generate embeddings for input text(s) with optimized batch processing.

        Converts text input into high-dimensional vector representations
        suitable for semantic similarity calculations and content analysis.
        Uses optimized inference mode for better performance.

        Args:
            sentences (Union[str, List[str]]): Single text string or list of texts
                to encode into embeddings
            batch_size (int): Number of texts to process in each batch.
                Larger batches are more efficient but use more memory.
                Defaults to EMBEDDING_MODEL_BATCH_SIZE.

        Returns:
            np.ndarray: Array of embeddings with shape (n_texts, embedding_dim).
                For single string input, shape is (1, embedding_dim).

        Raises:
            Exception: If encoding fails or invalid input provided

        """
        # Normalize input to list format
        if isinstance(sentences, str):
            sentences = [sentences]

        logger.debug(f"Encoding {len(sentences)} sentences with batch_size={batch_size}")

        try:
            # Validate input
            if not sentences:
                logger.warning("Empty sentences list provided")
                return np.array([]).reshape(0, EMBEDDING_VECTOR_SIZE)

            # Filter out empty or invalid sentences
            valid_sentences = [s for s in sentences if s and isinstance(s, str) and s.strip()]

            if len(valid_sentences) != len(sentences):
                logger.warning(f"Filtered {len(sentences) - len(valid_sentences)} invalid sentences")

            if not valid_sentences:
                logger.warning("No valid sentences to encode")
                return np.array([]).reshape(0, EMBEDDING_VECTOR_SIZE)

            # Generate embeddings with optimized inference
            with torch.inference_mode():
                logger.debug("Starting embedding generation")
                start_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
                end_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None

                if start_time:
                    start_time.record()

                embeddings = self.model.encode(
                    valid_sentences,
                    batch_size=batch_size,
                    device=self.device,
                    show_progress_bar=len(valid_sentences) > 100,  # Show progress for large batches
                    normalize_embeddings=True,  # L2 normalize for better similarity calculation
                )

                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                    logger.debug(f"Embedding generation took {elapsed_time:.2f} seconds")

            # Validate output
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            logger.debug(f"Embedding stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")

            # Handle case where some sentences were filtered out
            if len(valid_sentences) != len(sentences):
                # Create zero embeddings for invalid sentences
                full_embeddings = np.zeros((len(sentences), embeddings.shape[1]))
                valid_idx = 0
                for i, sentence in enumerate(sentences):
                    if sentence and isinstance(sentence, str) and sentence.strip():
                        full_embeddings[i] = embeddings[valid_idx]
                        valid_idx += 1
                embeddings = full_embeddings

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode sentences: {e}")
            # Return zero embeddings as fallback
            fallback_shape = (len(sentences), EMBEDDING_VECTOR_SIZE)
            logger.warning(f"Returning zero embeddings with shape: {fallback_shape}")
            return np.zeros(fallback_shape)

    def similarity(self, texts1: Union[str, List[str]], texts2: Union[str, List[str]]) -> np.ndarray:
        """
        Calculate cosine similarity between two sets of texts.

        Args:
            texts1: First set of texts
            texts2: Second set of texts

        Returns:
            np.ndarray: Similarity matrix with shape (len(texts1), len(texts2))
        """
        embeddings1 = self.encode(texts1)
        embeddings2 = self.encode(texts2)

        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(embeddings1, embeddings2)

        logger.debug(f"Calculated similarity matrix shape: {similarity_matrix.shape}")
        return similarity_matrix

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            dict: Dictionary containing model information
        """
        try:
            info = {
                "model_name": self.model_name,
                "model_path": str(self.model_path),
                "device": self.device,
                "embedding_dimension": getattr(
                    self.model, "get_sentence_embedding_dimension", lambda: EMBEDDING_VECTOR_SIZE
                )(),
                "max_seq_length": getattr(self.model, "max_seq_length", "unknown"),
                "cached": self.model_path.exists(),
            }

            logger.debug(f"Model info: {info}")
            return info

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        Make the embedder callable for convenient usage.

        This allows using the embedder instance as a function, which is
        more intuitive for many use cases.

        Args:
            *args: Arguments passed to encode method
            **kwargs: Keyword arguments passed to encode method

        Returns:
            np.ndarray: Embeddings array from encode method

        """
        return self.encode(*args, **kwargs)


if __name__ == "__main__":
    # Configure logging for main execution
    logging.basicConfig(level=logging.INFO)

    try:
        logger.info("Testing FeedShiftEmbeddor")

        # Initialize embedder
        emb = FeedShiftEmbeddor()

        # Test sentences
        sentence_list = ["AJ is good", "Nope, not good", "This is a test sentence"]

        # Generate embeddings
        logger.info(f"Encoding {len(sentence_list)} test sentences")
        embeddings = emb(sentence_list)

        # Display results
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Sample embedding norm: {np.linalg.norm(embeddings[0]):.4f}")

        # Test similarity
        similarity_matrix = emb.similarity(sentence_list[:2], sentence_list[1:])
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Sample similarity: {similarity_matrix[0, 0]:.4f}")

        logger.info("FeedShiftEmbeddor test completed successfully")

    except Exception as e:
        logger.error(f"Error during embedder test: {e}")
        raise
