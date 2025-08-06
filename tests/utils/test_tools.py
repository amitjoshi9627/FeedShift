import numpy as np
import pytest

from src.utils.tools import harmonic_mean


@pytest.mark.parametrize(
    "a, b, expected",
    [(np.array([1.0, 2.0]), np.array([2.0, 5.0]), np.array([1.3333333, 2.8571429])), (2.0, 5.0, 2.857142857142857)],
)
def test_harmonic_mean(a: np.ndarray | float, b: np.ndarray | float, expected: np.ndarray | float):
    result = harmonic_mean(a, b)
    np.testing.assert_almost_equal(result, expected)
