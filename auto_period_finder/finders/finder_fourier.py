from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Series

from auto_period_finder.tools import apply_window_fun, to_1d_array


class FourierPeriodFinder:
    """
    Fast Fourier Transform (FFT) based seasonality periods automatic finder.

    Find the periods of a given time series using FFT.

    Parameters
    ----------
    endog : array_like
        Data to be investigated. Must be squeezable to 1-d.

    See Also
    --------
    np.fft
        Discrete Fourier Transform.
    scipy.signal.get_window
        SciPy window function.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021)
    Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia.
    OTexts.com/fpp3/useful-predictors.html#fourier-series. Accessed on 05-22-2024.

    Examples
    --------
    Start by loading a timeseries dataset with a frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data

    You can resample the data to whatever frequency you want.

    >>> data = data.resample("ME").mean().ffill()

    Use FourierPeriodFinder to find the list of seasonality periods using FFT, ordered
    by corresponding frequency amplitudes in a descending order.

    >>> period_finder = FourierPeriodFinder(data)
    >>> periods = period_finder.fit()

    You can optionally specify a window function to pre-process with.

    >>> periods = period_finder.fit(window_func="blackman")
    """

    def __init__(self, endog: Union[ArrayLike, DataFrame, Series]):
        self.y = to_1d_array(endog)

    def fit(
        self,
        max_period_count: Optional[Union[int, None]] = None,
        window_func: Optional[Union[float, str, tuple, None]] = None,
    ) -> NDArray:
        """
        Find seasonality periods of the given time series automatically.

        Parameters
        ----------
        max_period_count : int, optional, default = None
            Maximum number of periods to look for.
        window_func : float, str, tuple optional, default = None
            Window function to be applied to the time series. Check
            'window' parameter documentation for scipy.signal.get_window
            function for more information on the accepted formats of this
            parameter.

        Returns
        -------
        list
            List of periods.
        """
        return self.__find_periods(
            self.y,
            max_period_count,
            window_func=window_func,
        )

    def __find_periods(
        self,
        y: Union[ArrayLike, DataFrame, Series],
        max_period_count: Optional[Union[int, None]],
        window_func: Optional[Union[str, float, tuple, None]],
    ) -> NDArray:
        # Apply window function on the series
        y = y if window_func is None else apply_window_fun(y, window_func=window_func)

        # Compute DFT and ignore the zero frequency
        freqs = np.fft.rfftfreq(len(y), d=1)[1:]
        ft = np.fft.rfft(y)[1:]

        # Compute periods and their respective amplitudes
        periods = np.round(1 / freqs)
        amps = abs(ft)

        # A period cannot be greater than half the length of the series
        result = pd.Series(index=periods, data=amps)[periods < len(y) // 2]

        # Return periods in descending order of amplitudes
        return (
            result.sort_values(ascending=False).index.unique().values[:max_period_count]
        )
