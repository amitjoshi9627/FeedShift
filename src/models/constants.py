"""
Machine learning model configuration and constants.

This module defines paths, parameters, and configuration values for
all ML models used in the FeedShift application, including embeddings,
toxicity detection, and other AI components.
"""

import logging
from pathlib import Path
from typing import Dict, Final

from src.config.paths import ROOT_DIR

# Configure logger
logger = logging.getLogger(__name__)

# Model storage configuration
MODEL_DIR: Final[Path] = ROOT_DIR / ".model_cache"
"""Base directory for storing all machine learning models and artifacts."""

# Embedding model configuration
EMBEDDING_MODEL_NAME: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
"""
HuggingFace model identifier for sentence embeddings.

This model provides good performance/speed tradeoff for semantic similarity tasks.
Alternative options: 'all-mpnet-base-v2' (higher quality, slower)
"""

EMBEDDING_MODEL_SAVED_NAME: Final[str] = "sentence_transformer"
"""Local directory name for the cached embedding model."""

EMBEDDING_MODEL_PATH: Final[Path] = MODEL_DIR / EMBEDDING_MODEL_SAVED_NAME
"""Full path to the locally cached embedding model."""

EMBEDDING_MODEL_BATCH_SIZE: Final[int] = 16
"""
Batch size for embedding model inference.

Larger batches improve throughput but require more GPU/CPU memory.
Adjust based on available hardware resources.
"""

EMBEDDING_VECTOR_SIZE: Final[int] = 384
"""Dimension size of the embedding vectors produced by the model."""

# Toxicity detection configuration
DEFAULT_TOXIC_SCORE: Final[float] = 0.8
"""
Default toxicity score assigned to content identified as toxic (0.0-1.0).

Higher values indicate more toxic content. This score is used when
toxicity detection models are unavailable or fail.
"""

DEFAULT_NON_TOXIC_SCORE: Final[float] = 0.1
"""
Default toxicity score assigned to content identified as non-toxic (0.0-1.0).

Lower values indicate cleaner content. This score represents the baseline
toxicity level for content that passes toxicity filters.
"""

TOXICITY_MODEL_NAME: Final[str] = "unitary/toxic-bert-base"
"""HuggingFace model identifier for toxicity detection."""

TOXICITY_MODEL_PATH: Final[Path] = MODEL_DIR / "toxicity_detector"
"""Local path for the cached toxicity detection model."""

TOXICITY_CONFIDENCE_THRESHOLD: Final[float] = 0.7
"""
Minimum confidence threshold for toxicity predictions (0.0-1.0).

Predictions below this threshold are treated with default scores.
"""

# Model performance configuration
MODEL_CACHE_SIZE: Final[int] = 1000
"""Maximum number of model predictions to cache in memory."""

MODEL_INFERENCE_TIMEOUT: Final[int] = 30
"""Timeout in seconds for model inference operations."""

# Device configuration
DEVICE_PREFERENCE: Final[str] = "auto"
"""
Device preference for model inference.

Options:
- "auto": Automatically select best available device
- "cpu": Force CPU usage
- "cuda": Force CUDA GPU usage (if available)
- "mps": Force Apple Metal Performance Shaders (if available)
"""

# Model downloading configuration
DOWNLOAD_TIMEOUT: Final[int] = 300
"""Timeout in seconds for model download operations."""

MAX_DOWNLOAD_RETRIES: Final[int] = 3
"""Maximum number of retry attempts for failed model downloads."""

# Quality and performance thresholds
MIN_TEXT_LENGTH_FOR_EMBEDDING: Final[int] = 5
"""Minimum text length (characters) required for embedding generation."""

MAX_TEXT_LENGTH_FOR_EMBEDDING: Final[int] = 500
"""Maximum text length (characters) for embedding generation (longer text is truncated)."""

SIMILARITY_THRESHOLD: Final[float] = 0.75
"""Threshold for considering two embeddings as similar (0.0-1.0)."""

# Model versioning and updates
MODEL_VERSION_CHECK_INTERVAL: Final[int] = 86400
"""Interval in seconds to check for model updates (24 hours)."""

ENABLE_MODEL_AUTO_UPDATE: Final[bool] = False
"""Whether to automatically download model updates when available."""


def ensure_model_directories() -> None:
    """
    Create all necessary model directories if they don't exist.

    This function should be called during model initialization to ensure
    all required directories are available for model storage.

    Raises:
        PermissionError: If unable to create directories due to permissions
        OSError: If directory creation fails for other reasons
    """
    directories = [
        MODEL_DIR,
        EMBEDDING_MODEL_PATH.parent,
        TOXICITY_MODEL_PATH.parent,
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured model directory exists: {directory}")
        except Exception as e:
            logger.error(f"Failed to create model directory {directory}: {e}")
            raise


def get_model_info() -> Dict[str, str]:
    """
    Get information about configured models.

    Returns:
        dict[str, str]: Dictionary containing model names and paths
    """
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_path": str(EMBEDDING_MODEL_PATH),
        "toxicity_model": TOXICITY_MODEL_NAME,
        "toxicity_path": str(TOXICITY_MODEL_PATH),
        "model_cache_dir": str(MODEL_DIR),
    }
