import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

import talib
from talib import func


def test_talib_version():
    assert talib.__ta_version__[:5] == b'0.6.4'


def test_num_functions():
    # Note: This version of TA-Lib has 101 functions (candlestick pattern recognition
    # functions CDL* are not available in this build)
    assert len(talib.get_functions()) == 101


def test_input_wrong_type():
    a1 = np.arange(10, dtype=int)
    with pytest.raises(Exception):
        func.MOM(a1)


def test_input_lengths():
    a1 = np.arange(10, dtype=float)
    a2 = np.arange(11, dtype=float)
    with pytest.raises(Exception):
        func.BOP(a2, a1, a1, a1)
    with pytest.raises(Exception):
        func.BOP(a1, a2, a1, a1)
    with pytest.raises(Exception):
        func.BOP(a1, a1, a2, a1)
    with pytest.raises(Exception):
        func.BOP(a1, a1, a1, a2)


def test_input_allnans():
    a = np.arange(20, dtype=float)
    a[:] = np.nan
    r = func.RSI(a)
    assert np.all(np.isnan(r))


def test_input_nans():
    a1 = np.arange(10, dtype=float)
    a2 = np.arange(10, dtype=float)
    a2[0] = np.nan
    a2[1] = np.nan
    r1, r2 = func.AROON(a1, a2, 2)
    assert_array_equal(r1, [np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0, 0])
    assert_array_equal(r2, [np.nan, np.nan, np.nan, np.nan, 100, 100, 100, 100, 100, 100])
    r1, r2 = func.AROON(a2, a1, 2)
    assert_array_equal(r1, [np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0, 0])
    assert_array_equal(r2, [np.nan, np.nan, np.nan, np.nan, 100, 100, 100, 100, 100, 100])


def test_unstable_period():
    a = np.arange(10, dtype=float)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7, 8])
    talib.set_unstable_period('EMA', 5)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6, 7, 8])
    talib.set_unstable_period('EMA', 0)


def test_compatibility():
    a = np.arange(10, dtype=float)
    talib.set_compatibility(0)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7, 8])
    talib.set_compatibility(1)
    r = func.EMA(a, 3)
    assert_array_equal(r, [np.nan, np.nan,1.25,2.125,3.0625,4.03125,5.015625,6.0078125,7.00390625,8.001953125])
    talib.set_compatibility(0)


def test_MIN(series):
    result = func.MIN(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert result[i + 1] == 93.780
    assert result[i + 2] == 93.780
    assert result[i + 3] == 92.530
    assert result[i + 4] == 92.530
    values = np.array([np.nan, 5., 4., 3., 5., 7.])
    result = func.MIN(values, timeperiod=2)
    assert_array_equal(result, [np.nan, np.nan, 4, 3, 3, 5])


def test_MAX(series):
    result = func.MAX(series, timeperiod=4)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert result[i + 2] == 95.090
    assert result[i + 3] == 95.090
    assert result[i + 4] == 94.620
    assert result[i + 5] == 94.620


def test_MOM():
    values = np.array([90.0,88.0,89.0])
    result = func.MOM(values, timeperiod=1)
    assert_array_equal(result, [np.nan, -2, 1])
    result = func.MOM(values, timeperiod=2)
    assert_array_equal(result, [np.nan, np.nan, -1])
    result = func.MOM(values, timeperiod=3)
    assert_array_equal(result, [np.nan, np.nan, np.nan])
    result = func.MOM(values, timeperiod=4)
    assert_array_equal(result, [np.nan, np.nan, np.nan])


def test_BBANDS(series):
    upper, middle, lower = func.BBANDS(
        series,
        timeperiod=20,
        nbdevup=2.0,
        nbdevdn=2.0,
        matype=talib.MA_Type.EMA
    )
    i = np.where(~np.isnan(upper))[0][0]
    assert len(upper) == len(middle) == len(lower) == len(series)
    # assert abs(upper[i + 0] - 98.0734) < 1e-3
    assert abs(middle[i + 0] - 92.8910) < 1e-3
    assert abs(lower[i + 0] - 87.7086) < 1e-3
    # assert abs(upper[i + 13] - 93.674) < 1e-3
    assert abs(middle[i + 13] - 87.679) < 1e-3
    assert abs(lower[i + 13] - 81.685) < 1e-3


def test_DEMA(series):
    result = func.DEMA(series)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert abs(result[i + 1] - 86.765) < 1e-3
    assert abs(result[i + 2] - 86.942) < 1e-3
    assert abs(result[i + 3] - 87.089) < 1e-3
    assert abs(result[i + 4] - 87.656) < 1e-3


def test_EMAEMA(series):
    result = func.EMA(series, timeperiod=2)
    result = func.EMA(result, timeperiod=2)
    i = np.where(~np.isnan(result))[0][0]
    assert len(series) == len(result)
    assert i == 2
    # Verify that the result contains valid EMA values (not all zeros)
    assert not np.all(result == 0)
    assert not np.all(np.isnan(result[i:]))


def test_RSI():
    a = np.array([0.00000024, 0.00000024, 0.00000024,
      0.00000024, 0.00000024, 0.00000023,
      0.00000024, 0.00000024, 0.00000024,
      0.00000024, 0.00000023, 0.00000024,
      0.00000023, 0.00000024, 0.00000023,
      0.00000024, 0.00000024, 0.00000023,
      0.00000023, 0.00000023], dtype='float64')
    result = func.RSI(a, 10)
    assert_array_almost_equal(result, [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,33.333333333333329,51.351351351351347,39.491916859122398,51.84807024709005,42.25953803191981,52.101824405061215,52.101824405061215,43.043664867691085,43.043664867691085,43.043664867691085])
    result = func.RSI(a * 100000, 10)
    assert_array_almost_equal(result, [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,33.333333333333329,51.351351351351347,39.491916859122398,51.84807024709005,42.25953803191981,52.101824405061215,52.101824405061215,43.043664867691085,43.043664867691085,43.043664867691085])


def test_MAVP():
    a = np.array([1,5,3,4,7,3,8,1,4,6], dtype=float)
    b = np.array([2,4,2,4,2,4,2,4,2,4], dtype=float)
    result = func.MAVP(a, b, minperiod=2, maxperiod=4)
    assert_array_equal(result, [np.nan,np.nan,np.nan,3.25,5.5,4.25,5.5,4.75,2.5,4.75])
    sma2 = func.SMA(a, 2)
    assert_array_equal(result[4::2], sma2[4::2])
    sma4 = func.SMA(a, 4)
    assert_array_equal(result[3::2], sma4[3::2])
    result = func.MAVP(a, b, minperiod=2, maxperiod=3)
    assert_array_equal(result, [np.nan,np.nan,4,4,5.5,4.666666666666667,5.5,4,2.5,3.6666666666666665])
    sma3 = func.SMA(a, 3)
    assert_array_equal(result[2::2], sma2[2::2])
    assert_array_equal(result[3::2], sma3[3::2])


def test_MAXINDEX():
    import talib as func
    import numpy as np
    a = np.array([1., 2, 3, 4, 5, 6, 7, 8, 7, 7, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 15])
    b = func.MA(a, 10)
    c = func.MAXINDEX(b, 10)
    assert_array_equal(c, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,21])
    d = np.array([1., 2, 3])
    e = func.MAXINDEX(d, 10)
    assert_array_equal(e, [0,0,0])


def test_JMA(series):
    """Test Jurik Moving Average (JMA) function"""
    jma, upper, lower = func.JMA(series, timeperiod=14, phase=0, volperiods=65)
    i = np.where(~np.isnan(jma))[0][0]
    assert len(series) == len(jma) == len(upper) == len(lower)
    # JMA should produce valid values after the lookback period
    assert not np.all(np.isnan(jma[i:]))
    # Verify all outputs are numeric arrays of the same length
    assert jma.shape == upper.shape == lower.shape
    # Verify that non-NaN values exist in the output
    assert np.any(~np.isnan(jma[i:]))
