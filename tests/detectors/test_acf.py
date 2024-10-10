from statsmodels.datasets import co2

from pyriodicity import ACFPeriodicityDetector


def test_co2_daily_acf_default():
    data = co2.load().data.resample("D").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit()
    assert len(periods) > 0


def test_co2_weekly_acf_default():
    data = co2.load().data.resample("W").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit()
    assert len(periods) > 0


def test_co2_monthly_acf_default():
    data = co2.load().data.resample("ME").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit()
    assert len(periods) > 0


def test_co2_daily_acf_max_period_count_one():
    data = co2.load().data.resample("D").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1)
    assert len(periods) == 1
    assert 364 in periods


def test_co2_weekly_acf_max_period_count_one():
    data = co2.load().data.resample("W").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1)
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_acf_max_period_count_one():
    data = co2.load().data.resample("ME").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1)
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_acf_max_period_count_one_correlation_func_spearman():
    data = co2.load().data.resample("D").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="spearman")
    assert len(periods) == 1
    assert 364 in periods


def test_co2_weekly_acf_max_period_count_one_correlation_func_spearman():
    data = co2.load().data.resample("W").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="spearman")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_acf_max_period_count_one_correlation_func_spearman():
    data = co2.load().data.resample("ME").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="spearman")
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_acf_max_period_count_one_correlation_func_kendall():
    data = co2.load().data.resample("D").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="kendall")
    assert len(periods) == 1
    assert 364 in periods


def test_co2_weekly_acf_max_period_count_one_correlation_func_kendall():
    data = co2.load().data.resample("W").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="kendall")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_acf_max_period_count_one_correlation_func_kendall():
    data = co2.load().data.resample("ME").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="kendall")
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_acf_max_period_count_one_window_func_blackman():
    data = co2.load().data.resample("D").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert 364 in periods


def test_co2_weekly_acf_max_period_count_one_window_func_blackman():
    data = co2.load().data.resample("W").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_acf_max_period_count_one_window_func_blackman():
    data = co2.load().data.resample("ME").mean().ffill()
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert 12 in periods
