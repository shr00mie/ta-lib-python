import numpy as np
import pandas as pd

from talib import stream


def test_streaming():
    a = np.array([1,1,2,3,5,8,13], dtype=float)
    r = stream.MOM(a, timeperiod=1)
    assert r == 5
    r = stream.MOM(a, timeperiod=2)
    assert r == 8
    r = stream.MOM(a, timeperiod=3)
    assert r == 10
    r = stream.MOM(a, timeperiod=4)
    assert r == 11
    r = stream.MOM(a, timeperiod=5)
    assert r == 12
    r = stream.MOM(a, timeperiod=6)
    assert r == 12
    r = stream.MOM(a, timeperiod=7)
    assert np.isnan(r)


def test_streaming_pandas():
    a = pd.Series([1,1,2,3,5,8,13])
    r = stream.MOM(a, timeperiod=1)
    assert r == 5
    r = stream.MOM(a, timeperiod=2)
    assert r == 8
    r = stream.MOM(a, timeperiod=3)
    assert r == 10
    r = stream.MOM(a, timeperiod=4)
    assert r == 11
    r = stream.MOM(a, timeperiod=5)
    assert r == 12
    r = stream.MOM(a, timeperiod=6)
    assert r == 12
    r = stream.MOM(a, timeperiod=7)
    assert np.isnan(r)


def test_MAXINDEX():
    a = np.array([1., 2, 3, 4, 5, 6, 7, 8, 7, 7, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 15])
    r = stream.MAXINDEX(a, 10)
    assert r == 21
