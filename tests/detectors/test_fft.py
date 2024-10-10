from statsmodels.datasets import co2

from pyriodicity import FFTPeriodicityDetector


def test_co2_monthly_fft_find_all_periods():
    data = co2.load().data.resample("ME").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit()
    assert len(periods) > 0


def test_co2_monthly_fft_find_first_two_periods():
    data = co2.load().data.resample("ME").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=2)
    assert len(periods) == 2


def test_co2_daily_fft_find_strongest_period():
    data = co2.load().data.resample("D").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1)
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period():
    data = co2.load().data.resample("W").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1)
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period():
    data = co2.load().data.resample("ME").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1)
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_barthann():
    data = co2.load().data.resample("D").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="barthann")
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_barthann():
    data = co2.load().data.resample("W").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="barthann")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_barthann():
    data = co2.load().data.resample("ME").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="barthann")
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_bartlett():
    data = co2.load().data.resample("D").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="bartlett")
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_bartlett():
    data = co2.load().data.resample("W").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="bartlett")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_bartlett():
    data = co2.load().data.resample("ME").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="bartlett")
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_blackman():
    data = co2.load().data.resample("D").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_blackman():
    data = co2.load().data.resample("W").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_blackman():
    data = co2.load().data.resample("ME").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_blackmanharris():
    data = co2.load().data.resample("D").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="blackmanharris")
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_blackmanharris():
    data = co2.load().data.resample("W").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="blackmanharris")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_blackmanharris():
    data = co2.load().data.resample("ME").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="blackmanharris")
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_boxcar():
    data = co2.load().data.resample("D").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="boxcar")
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_boxcar():
    data = co2.load().data.resample("W").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="boxcar")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_boxcar():
    data = co2.load().data.resample("ME").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="boxcar")
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_hamming():
    data = co2.load().data.resample("D").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="hamming")
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_hamming():
    data = co2.load().data.resample("W").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="hamming")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_hamming():
    data = co2.load().data.resample("ME").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="hamming")
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_hann():
    data = co2.load().data.resample("D").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="hann")
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_hann():
    data = co2.load().data.resample("W").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="hann")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_hann():
    data = co2.load().data.resample("ME").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="hann")
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_tukey():
    data = co2.load().data.resample("D").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="tukey")
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_tukey():
    data = co2.load().data.resample("W").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="tukey")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_tukey():
    data = co2.load().data.resample("ME").mean().ffill()
    fft_detector = FFTPeriodicityDetector(data)
    periods = fft_detector.fit(max_period_count=1, window_func="tukey")
    assert len(periods) == 1
    assert 12 in periods
