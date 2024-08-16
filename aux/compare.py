"""!
    Compare Module
    ==============

    This module implements functions for the comparison of FIs computed using different methds.

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import dataclasses

import numpy as np
from sklearn import metrics


@dataclasses.dataclass
class ComparisonMetrics:
    """!Comparison metrics for FIs"""

    mad: np.ndarray
    rho: np.ndarray
    r2: np.ndarray
    names: list[str]


def standardize(x: np.ndarray) -> np.ndarray:
    """!Standardize a vector/time series

    @param x The vector/time series to be standardized.
    @return The standardized vector/time series.
    """
    return (x - np.nanmean(x)) / np.nanstd(x)


def resample_to_n_samples(x: np.ndarray, n: int) -> np.ndarray:
    """!Resample a vector/time series to a specified number of samples

    @param x The input vector/time series to be resampled.
    @param n The desired number of samples in the output.
    @return The resampled vector/time series with n samples.
    """
    original_length = len(x)
    if original_length == n:
        return x

    t = np.linspace(0, 1, n)
    tp = np.linspace(0, 1, original_length)
    return np.interp(t, tp, x)


def compare_signals(xs: np.ndarray, names: list[str]) -> ComparisonMetrics:
    """!Compare signals

    @param xs The signals to be compared: each _row_ is a signal/variable
    @param names The names of the signals.
    @return TBD
    """
    n = len(names)
    rho = np.corrcoef(xs)
    r2 = np.zeros((n, n))
    mad = np.zeros((n, n))
    for ii in range(n):
        for jj in range(ii + 1, n):
            r2[ii, jj] = metrics.r2_score(xs[ii], xs[jj])
            mad[ii, jj] = metrics.mean_absolute_error(xs[ii], xs[jj])

    r2 += r2.T
    mad += mad.T
    np.fill_diagonal(r2, 1.0)

    return ComparisonMetrics(mad, rho, r2, names)
