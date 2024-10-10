from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import argrelmax, butter, detrend, periodogram, sosfiltfilt

from pyriodicity.tools import acf, apply_window, power_threshold, to_1d_array


class CFDAutoperiod:
    """
    CFDAutoperiod periodicity detector.

    Find the periods in a given signal or series using CFDAutoperiod [1]_.

    Parameters
    ----------
    endog : array_like
        Data to be investigated. Must be squeezable to 1-d.

    References
    ----------
    .. [1] Puech, T., Boussard, M., D'Amato, A., & Millerand, G. (2020).
       A fully automated periodicity detection in time series. In Advanced
       Analytics and Learning on Temporal Data: 4th ECML PKDD Workshop, AALTD 2019,
       Würzburg, Germany, September 20, 2019, Revised Selected Papers 4 (pp. 43-54).
       Springer International Publishing. https://doi.org/10.1007/978-3-030-39098-3_4

    Examples
    --------
    Start by loading a timeseries datasets and resampling to an appropriate
    frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data
    >>> data = data.resample("ME").mean().ffill()

    Use ``CFDAutoperiod`` to find the list of periods in the data.

    >>> from pyriodicity import CFDAutoperiod
    >>> cfd_autoperiod = CFDAutoperiod(data)
    >>> cfd_autoperiod.fit()
    array([12])

    You can specify a lower percentile value should you wish for
    a more lenient detection

    >>> cfd_autoperiod.fit(percentile=90)
    array([12])

    You can also increase the number of random data permutations
    for a more robust power threshold estimation

    >>> cfd_autoperiod.fit(k=300)
    array([12])

    ``CFDAutoperiod`` is considered a more robust variant of ``Autoperiod``.
    The detection algorithm found exactly one periodicity of 12, suggesting
    a strong yearly periodicity.
    """

    def __init__(self, endog: ArrayLike):
        self.y = to_1d_array(endog)

    def fit(
        self,
        k: int = 100,
        percentile: int = 99,
        detrend_func: Optional[str] = "linear",
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
        percentile : int, optional, default = 99
            Percentage for the percentile parameter used in computing the power
            threshold. Value must be between 0 and 100 inclusive.
        detrend_func : str, default = 'linear'
            The kind of detrending to be applied on the signal. It can either be
            'linear' or 'constant'.
        window_func : float, str, tuple optional, default = None
            Window function to be applied to the time series. Check
            'window' parameter documentation for ``scipy.signal.get_window``
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
        self.y = self.y if detrend_func is None else detrend(self.y, type=detrend_func)
        # Apply window on data
        self.y = self.y if window_func is None else apply_window(self.y, window_func)

        # Compute the power threshold
        p_threshold = power_threshold(self.y, detrend_func, k, percentile)

        # Find period hints
        freq, power = periodogram(self.y, detrend=detrend_func)
        hints = np.array(
            [
                1 / f
                for f, p in zip(freq, power)
                if f >= 1 / len(freq) and p >= p_threshold
            ]
        )

        # Replace period hints with their density clustering centroids
        hints = self._cluster_period_hints(hints, len(self.y))

        # Validate period hints
        valid_hints = []
        length = len(self.y)
        y_filtered = np.array(self.y)
        for h in hints:
            if self._is_hint_valid(y_filtered, h, detrend_func, correlation_func):
                # Apply a low pass filter with an adapted cutoff frequency for the next hint
                f_cuttoff = 1 / (length / (length / h + 1) - 1)
                y_filtered = sosfiltfilt(
                    butter(N=5, Wn=f_cuttoff, output="sos"), y_filtered
                )
                valid_hints.append(h)

        # Calculate only the needed part of the ACF array for each hint
        hint_ranges = [
            np.arange(h // 2, 1 + h + h // 2, dtype=int) for h in valid_hints
        ]
        acf_arrays = [
            acf(
                self.y,
                lag_start=r[0],
                lag_stop=r[-1],
                correlation_func=correlation_func,
            )
            for r in hint_ranges
        ]

        # Return the closest ACF peak to each valid period hint
        return np.array(
            [
                r[0] + min(argrelmax(arr)[0], key=lambda x: abs(x - h))
                for h, r, arr in zip(valid_hints, hint_ranges, acf_arrays)
            ]
        )

    @staticmethod
    def _cluster_period_hints(period_hints: ArrayLike, n: int) -> NDArray:
        """
        Find the centroids of the period hint density clusters.

        Parameters
        ----------
        period_hints : array_like
            List of period hints.
        n : int
            Length of the data.

        Returns
        -------
        NDArray
            List of period hint density cluster centroids.
        """
        hints = np.sort(period_hints)
        eps = [
            hints[i] if i == 0 else 1 + n / (n / hints[i - 1] - 1)
            for i in range(len(hints))
        ]
        clusters = np.split(hints, np.argwhere(hints > eps).flatten())
        return np.array([c.mean() for c in clusters if len(c) > 0])

    @staticmethod
    def _is_hint_valid(
        y: ArrayLike,
        hint: float,
        detrend_func: Union[str],
        correlation_func: str,
    ) -> bool:
        """
        Validate the period hint.

        Parameters
        ----------
        y : array_like
            Data to be investigated. Must be squeezable to 1-d.
        hint : float
            The period hint to be validated.
        detrend_func : str
            The kind of detrending to be applied on the signal. It can either be
            'linear' or 'constant'.
        correlation_func : str
            The correlation function to be used to calculate the ACF of the series
            or the signal. Possible values are ['pearson', 'spearman', 'kendall'].

        Returns
        -------
        bool
            Whether the period hint is valid.
        """
        hint_range = np.arange(hint // 2, 1 + hint + hint // 2, dtype=int)
        acf_arr = acf(
            y,
            lag_start=hint_range[0],
            lag_stop=hint_range[-1],
            correlation_func=correlation_func,
        )
        polynomial = np.polynomial.Polynomial.fit(
            hint_range, detrend(acf_arr, type=detrend_func), deg=2
        ).convert()
        derivative = polynomial.deriv()
        return polynomial.coef[-1] < 0 and int(derivative.roots()[0]) in hint_range
