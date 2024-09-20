from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import argrelmax, periodogram
from scipy.stats import linregress

from pyriodicity.tools import acf, apply_window, detrend, to_1d_array


class Autoperiod:
    """
    Autoperiod periodicity detector.

    Find the periods in a given signal or series using Autoperiod [1]_.

    Parameters
    ----------
    endog : array_like
        Data to be investigated. Must be squeezable to 1-d.

    References
    ----------
    .. [1] Vlachos, M., Yu, P., & Castelli, V. (2005).
       On periodicity detection and Structural Periodic similarity.
       Proceedings of the 2005 SIAM International Conference on Data Mining.
       https://doi.org/10.1137/1.9781611972757.40

    Examples
    --------
    Start by loading a timeseries datasets and resampling to an appropriate
    frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data
    >>> data = data.resample("ME").mean().ffill()

    Use ``Autoperiod`` to find the list of periods in the data.

    >>> from pyriodicity import Autoperiod
    >>> autoperiod = Autoperiod(data)
    >>> autoperiod.fit()
    array([12])

    You can specify a lower percentile value should you wish for
    a more lenient detection

    >>> autoperiod.fit(percentile=90)
    array([12])

    You can also increase the number of random data permutations
    for a more robust power threshold estimation

    >>> autoperiod.fit(k=300)
    array([12])

    ``Autoperiod`` is generally a quite robust periodicity detection method.
    The detection algorthim found exactly one periodicity of 12, suggesting
    a strong yearly periodicity.
    """

    def __init__(self, endog: ArrayLike):
        self.y = to_1d_array(endog)

    def fit(
        self,
        k: int = 100,
        percentile: int = 95,
        detrend_func: Optional[Union[str, Callable[[ArrayLike], NDArray]]] = "linear",
        window_func: Optional[Union[str, float, tuple]] = None,
        correlation_func: Optional[str] = "pearson",
    ) -> NDArray:
        """
        Find periods in the given series.

        Parameters
        ----------
        k : int, optional, default = 100
            The number of times the data is randomly permuted while estimating the
            power threshold.
        percentile : int, optional, default = 95
            Percentage for the percentile parameter used in computing the power
            threshold. Value must be between 0 and 100 inclusive.
        detrend_func : str, callable, default = None
            The kind of detrending to be applied on the series. It can either be
            'linear' or 'constant' if it the parameter is of 'str' type, or a
            custom function that returns a detrended series.
        window_func : float, str, tuple optional, default = None
            Window function to be applied to the time series. Check
            'window' parameter documentation for scipy.signal.get_window
            function for more information on the accepted formats of this
            parameter.
        correlation_func : str, default = 'pearson'
            The correlation function to be used to calculate the ACF of the time
            series. Possible values are ['pearson', 'spearman', 'kendall'].

        Returns
        -------
        NDArray
            List of detected periods.

        See Also
        --------
        scipy.signal.detrend
            Remove linear trend along axis from data.
        scipy.signal.get_window
            Return a window of a given length and type.
        scipy.stats.kendalltau
            Calculate Kendall's tau, a correlation measure for ordinal data.
        scipy.stats.pearsonr
            Pearson correlation coefficient and p-value for testing non-correlation.
        scipy.stats.spearmanr
            Calculate a Spearman correlation coefficient with associated p-value.

        """
        # Detrend data
        self.y = self.y if detrend_func is None else detrend(self.y, detrend_func)
        # Apply window on data
        self.y = self.y if window_func is None else apply_window(self.y, window_func)

        # Compute the power threshold
        p_threshold = self._power_threshold(self.y, k, percentile)

        # Find period hints
        freq, power = periodogram(self.y, window=None, detrend=None)
        period_hints = np.array(
            [
                1 / f
                for f, p in zip(freq, power)
                if f >= 1 / len(freq) and p >= p_threshold
            ]
        )

        # Compute the ACF
        length = len(self.y)
        acf_arr = acf(self.y, nlags=length, correlation_func=correlation_func)

        # Validate period hints
        period_hints_valid = []
        for p in period_hints:
            q = length / p
            start = np.floor((p + length / (q + 1)) / 2 - 1).astype(int)
            end = np.ceil((p + length / (q - 1)) / 2 + 1).astype(int)

            splits = [
                self._split(np.arange(len(acf_arr)), acf_arr, start, end, i)
                for i in range(start + 2, end)
            ]
            line1, line2, _ = splits[
                np.array([error for _, _, error in splits]).argmin()
            ]

            if line1.slope > 0 > line2.slope:
                period_hints_valid.append(p)

        period_hints_valid = np.array(period_hints_valid)

        # Return the closest ACF peak for each valid period hint
        local_argmax = argrelmax(acf_arr)[0]
        return np.array(
            list(
                {
                    min(local_argmax, key=lambda x: abs(x - p))
                    for p in period_hints_valid
                }
            )
        )

    @staticmethod
    def _power_threshold(y: ArrayLike, k: int, p: int) -> float:
        """
        Compute the power threshold as the p-th percentile of the maximum
        power values of the periodogram of k permutations of the data.

        Parameters
        ----------
        y : array_like
            Data to be investigated. Must be squeezable to 1-d.
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
        max_powers = []
        while len(max_powers) < k:
            _, power_p = periodogram(
                np.random.permutation(y), window=None, detrend=None
            )
            max_powers.append(power_p.max())
        max_powers.sort()
        return np.percentile(max_powers, p)

    @staticmethod
    def _split(x: ArrayLike, y: ArrayLike, start: int, end: int, split: int) -> tuple:
        """
        Approximate a function at [start, end] with two line segments at
        [start, split - 1] and [split, end].

        Parameters
        ----------
        x : array_like
            The x-coordinates of the data points.
        y : array_like
            The y-coordinates of the data points.
        start : int
            The start index of the data points to be approximated.
        end : int
            The end index of the data points to be approximated.
        split : int
            The split index of the data points to be approximated.

        See Also
        --------
        scipy.stats.linregress
            Calculate a linear least-squares regression for two sets of measurements.

        Returns
        -------
        linregress
            The first line segment.
        linregress
            The second line segment.
        float
            The error of the approximation.
        """
        x1, y1, x2, y2 = (
            x[start:split],
            y[start:split],
            x[split : end + 1],
            y[split : end + 1],
        )
        line1 = linregress(x1, y1)
        line2 = linregress(x2, y2)
        error = np.sum(np.abs(y1 - (line1.intercept + line1.slope * x1))) + np.sum(
            np.abs(y2 - (line2.intercept + line2.slope * x2))
        )
        return line1, line2, error
