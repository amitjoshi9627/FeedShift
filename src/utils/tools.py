"""
Utility functions for data processing and mathematical operations.

This module provides common utility functions used throughout the FeedShift
application for file I/O operations, mathematical calculations, and data
processing tasks.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with error handling.

    Provides robust CSV loading with comprehensive error handling,
    logging, and validation. Supports both string and Path objects
    for file paths.

    Args:
        path (Union[str, Path]): Path to the CSV file to load

    Returns:
        pd.DataFrame: Loaded DataFrame from the CSV file

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
        pd.errors.ParserError: If the CSV file is malformed
        Exception: For other unexpected errors during loading

    Example:
        >>> df = load_csv('data/posts.csv')
        >>> print(f"Loaded {len(df)} records")
    """
    path = Path(path)
    logger.debug(f"Loading CSV file: {path}")

    try:
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        # Check file size
        file_size = path.stat().st_size
        if file_size == 0:
            logger.warning(f"CSV file is empty: {path}")
            return pd.DataFrame()

        # Load CSV with optimized settings
        df = pd.read_csv(
            path,
            encoding="utf-8",
            na_values=["", "NULL", "null", "None", "N/A"],
            keep_default_na=True,
            low_memory=False,  # Better type inference
        )

        logger.info(f"Successfully loaded CSV: {len(df)} rows, {len(df.columns)} columns from {path}")
        logger.debug(f"CSV columns: {list(df.columns)}")

        # Check for completely empty DataFrame
        if df.empty:
            logger.warning("Loaded DataFrame is empty")

        return df

    except FileNotFoundError:
        logger.error(f"CSV file not found: {path}")
        raise
    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file is empty: {path}")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading CSV {path}: {e}")
        raise


def save_csv(data: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save a pandas DataFrame to a CSV file with error handling.

    Provides robust CSV saving with directory creation, error handling,
    and validation. Ensures the target directory exists before saving.

    Args:
        data (pd.DataFrame): DataFrame to save to CSV
        path (Union[str, Path]): Destination path for the CSV file

    Raises:
        ValueError: If the DataFrame is None or invalid
        PermissionError: If unable to write to the specified path
        Exception: For other unexpected errors during saving

    Example:
        >>> save_csv(df, 'output/processed_data.csv')
        >>> print("Data saved successfully")
    """
    path = Path(path)
    logger.debug(f"Saving CSV file: {path}")

    try:
        # Validate input DataFrame
        if data is None:
            raise ValueError("Cannot save None DataFrame")

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(data)}")

        # Log DataFrame info
        logger.debug(f"Saving DataFrame: {len(data)} rows, {len(data.columns)} columns")

        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path.parent}")

        # Save with optimized settings
        data.to_csv(
            path,
            index=False,
            encoding="utf-8",
            na_rep="",  # How to represent NaN values
            float_format="%.6g",  # Compact float representation
        )

        # Verify save was successful
        if path.exists():
            file_size = path.stat().st_size
            logger.info(f"Successfully saved CSV: {len(data)} rows to {path} ({file_size} bytes)")
        else:
            logger.error(f"CSV save verification failed: {path}")
            raise Exception("File was not created successfully")

    except ValueError as e:
        logger.error(f"Invalid input for CSV save: {e}")
        raise
    except PermissionError as e:
        logger.error(f"Permission denied saving CSV to {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving CSV to {path}: {e}")
        raise


def harmonic_mean(a: Union[np.ndarray, float], b: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Calculate the harmonic mean of two arrays or scalars.

    The harmonic mean is useful for averaging rates and ratios, providing
    a more conservative average than arithmetic mean. It's particularly
    useful in ranking systems where you want to balance multiple factors.

    The harmonic mean is calculated as: 2 * a * b / (a + b)

    Args:
        a (Union[np.ndarray, float]): First array or scalar value
        b (Union[np.ndarray, float]): Second array or scalar value

    Returns:
        Union[np.ndarray, float]: Harmonic mean of the inputs.
            Returns 0 if either input contains all zeros.
            Same type as inputs (array or scalar).

    Raises:
        ValueError: If inputs have incompatible shapes
        TypeError: If inputs are not numeric types

    Example:
        >>> result = harmonic_mean(0.8, 0.6)
        >>> print(result)  # 0.6857 (approximately)

        >>> scores1 = np.array([0.1, 0.5, 0.9])
        >>> scores2 = np.array([0.2, 0.4, 0.8])
        >>> result = harmonic_mean(scores1, scores2)
        >>> print(result)  # [0.133, 0.444, 0.842]
    """
    logger.debug(f"Calculating harmonic mean for inputs with types: {type(a)}, {type(b)}")

    try:
        # Convert to numpy arrays for consistent handling
        a_array = np.asarray(a)
        b_array = np.asarray(b)

        # Validate input types
        if not np.issubdtype(a_array.dtype, np.number):
            raise TypeError(f"Input 'a' must be numeric, got {a_array.dtype}")
        if not np.issubdtype(b_array.dtype, np.number):
            raise TypeError(f"Input 'b' must be numeric, got {b_array.dtype}")

        # Check for shape compatibility
        try:
            # This will raise an error if shapes are incompatible
            np.broadcast_arrays(a_array, b_array)
        except ValueError as e:
            raise ValueError(f"Incompatible input shapes: {a_array.shape} and {b_array.shape}") from e

        # Handle zero cases
        # If both are zero arrays/scalars, return zero
        if np.all(a_array == 0) and np.all(b_array == 0):
            logger.debug("Both inputs are zero, returning zero")
            return type(a)(0) if np.isscalar(a) else np.zeros_like(a_array)

        # If either is zero, return zero
        if np.all(a_array == 0) or np.all(b_array == 0):
            logger.debug("One input is zero, returning zero")
            return type(a)(0) if np.isscalar(a) else np.zeros_like(a_array)

        # Calculate harmonic mean: 2 * a * b / (a + b)
        numerator = 2 * a_array * b_array
        denominator = a_array + b_array

        # Handle division by zero (shouldn't happen if inputs are valid)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0))

        # Convert back to original type if input was scalar
        if np.isscalar(a) and np.isscalar(b):
            result = result.item()

        logger.debug("Harmonic mean calculated successfully")
        return result

    except Exception as e:
        logger.error(f"Error calculating harmonic mean: {e}")
        raise


def validate_dataframe(df: pd.DataFrame, required_columns: list | None = None) -> bool:
    """
    Validate a DataFrame has required structure and content.

    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present

    Returns:
        bool: True if DataFrame is valid

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df)}")

    if df.empty:
        raise ValueError("DataFrame is empty")

    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    logger.debug(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def normalize_array(arr: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize an array using specified method.

    Args:
        arr (np.ndarray): Array to normalize
        method (str): Normalization method ('minmax', 'zscore', 'l2')

    Returns:
        np.ndarray: Normalized array
    """
    if method == "minmax":
        min_val, max_val = arr.min(), arr.max()
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    elif method == "zscore":
        mean_val, std_val = arr.mean(), arr.std()
        if std_val == 0:
            return np.zeros_like(arr)
        return (arr - mean_val) / std_val

    elif method == "l2":
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr
        return arr / norm

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def safe_divide(
    numerator: Union[np.ndarray, float], denominator: Union[np.ndarray, float], default: float = 0.0
) -> Union[np.ndarray, float]:
    """
    Perform safe division with handling for division by zero.

    Args:
        numerator: Numerator values
        denominator: Denominator values
        default: Value to return when denominator is zero

    Returns:
        Division result with safe handling of zero denominators
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(
            numerator, denominator, out=np.full_like(numerator, default, dtype=float), where=(denominator != 0)
        )
    return result
