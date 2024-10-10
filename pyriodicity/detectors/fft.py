from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import detrend

from pyriodicity.tools import apply_window, to_1d_array


class FFTPeriodicityDetector:
    """
    Fast Fourier Transform (FFT) based periodicity detector.

    Find the periods in a given signal or series using FFT.

    Parameters
    ----------
    endog : array_like
        Data to be investigated. Must be squeezable to 1-d.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021)
    Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia.
    OTexts.com/fpp3/useful-predictors.html#fourier-series. Accessed on 09-15-2024.

    Examples
    --------
    Start by loading a timeseries dataset.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data

    You can resample the data to whatever frequency you want.

    >>> data = data.resample("ME").mean().ffill()

    Use FFTPeriodicityDetector to find the list of periods using FFT, ordered
    by corresponding frequency amplitudes in a descending order.

    >>> fft_detector = FFTPeriodicityDetector(data)
    >>> periods = fft_detector.fit()

    You can optionally specify a window function for pre-processing.

    >>> periods = fft_detector.fit(window_func="blackman")
    """

    def __init__(self, endog: ArrayLike):
        self.y = to_1d_array(endog)

    def fit(
        self,
        max_period_count: Optional[int] = None,
        detrend_func: Optional[str] = "linear",
        window_func: Optional[Union[float, str, tuple]] = None,
    ) -> NDArray:
        """
        Find periods in the given series.

        Parameters
        ----------
        max_period_count : int, optional, default = None
            Maximum number of periods to look for.
        detrend_func : str, default = 'linear'
            The kind of detrending to be applied on the signal. It can either be
            'linear' or 'constant'.
        window_func : float, str, tuple optional, default = None
            Window function to be applied to the time series. Check
            'window' parameter documentation for scipy.signal.get_window
            function for more information on the accepted formats of this
            parameter.

        See Also
        --------
        numpy.fft
            Discrete Fourier Transform.
        scipy.signal.detrend
            Remove linear trend along axis from data.
        scipy.signal.get_window
            Return a window of a given length and type.

        Returns
        -------
        NDArray
            List of detected periods.
        """
        # Detrend data
        self.y = self.y if detrend_func is None else detrend(self.y, type=detrend_func)

        # Apply the window function on the data
        self.y = (
            self.y
            if window_func is None
            else apply_window(self.y, window_func=window_func)
        )

        # Compute DFT and ignore the zero frequency
        freqs = np.fft.rfftfreq(len(self.y), d=1)[1:]
        ft = np.fft.rfft(self.y)[1:]

        # Compute period lengths and their respective amplitudes
        periods = np.round(1 / freqs).astype(int)
        amps = abs(ft)

        # A period cannot be greater than half the length of the series
        filter = periods < len(self.y) // 2
        periods = periods[filter]
        amps = amps[filter]

        # Sort period length values in the descending order of their corresponding amplitudes
        periods = periods[np.argsort(-amps)]

        # Return unique period length values
        _, unique_indices = np.unique(periods, return_index=True)
        return periods[np.sort(unique_indices)][:max_period_count]
