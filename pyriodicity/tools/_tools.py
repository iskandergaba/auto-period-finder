from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window, periodogram
from scipy.stats import kendalltau, pearsonr, spearmanr


@staticmethod
def to_1d_array(x: ArrayLike) -> NDArray:
    y = np.ascontiguousarray(np.squeeze(np.asarray(x)), dtype=np.double)
    if y.ndim != 1:
        raise ValueError("y must be a 1d array")
    return y


@staticmethod
def apply_window(x: ArrayLike, window_func: Union[str, float, tuple]) -> NDArray:
    return x * get_window(window=window_func, Nx=len(x))


@staticmethod
def acf(
    x: ArrayLike,
    lag_start: int,
    lag_stop: int,
    correlation_func: Optional[str] = "pearson",
) -> NDArray:
    if not 0 <= lag_start < lag_stop <= len(x):
        raise ValueError(
            "Invalid lag values range ({}, {})".format(lag_start, lag_stop)
        )
    lag_values = np.arange(lag_start, lag_stop + 1, dtype=int)
    if correlation_func == "spearman":
        return np.array([spearmanr(x, np.roll(x, l)).statistic for l in lag_values])
    elif correlation_func == "kendall":
        return np.array([kendalltau(x, np.roll(x, l)).statistic for l in lag_values])
    return np.array([pearsonr(x, np.roll(x, l)).statistic for l in lag_values])


@staticmethod
def power_threshold(
    y: ArrayLike,
    detrend_func: str,
    k: int,
    p: int,
) -> float:
    """
    Compute the power threshold as the p-th percentile of the maximum
    power values of the periodogram of k permutations of the data.

    Parameters
    ----------
    y : array_like
        Data to be investigated. Must be squeezable to 1-d.
    detrend_func : str, default = 'linear'
        The kind of detrending to be applied on the series. It can either be
        'linear' or 'constant'.
    k : int
        The number of times the data is randomly permuted to compute
        the maximum power values.
    p : int
        The percentile value used to compute the power threshold.
        It determines the cutoff point in the sorted list of the maximum
        power values from the periodograms of the permuted data.
        Value must be between 0 and 100 inclusive.

    See Also
    --------
    scipy.signal.periodogram
        Estimate power spectral density using a periodogram.

    Returns
    -------
    float
        Power threshold of the target data.
    """
    if detrend_func is None:
        detrend_func = False
    max_powers = []
    while len(max_powers) < k:
        _, power_p = periodogram(np.random.permutation(y), detrend=detrend_func)
        max_powers.append(power_p.max())
    max_powers.sort()
    return np.percentile(max_powers, p)
