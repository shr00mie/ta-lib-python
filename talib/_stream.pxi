cannot find defaults and docs for ACCBANDS
cannot find defaults and docs for ACOS
cannot find defaults and docs for AD
cannot find defaults and docs for ADD
cannot find defaults and docs for ADOSC
cannot find defaults and docs for ADX
cannot find defaults and docs for ADXR
cannot find defaults and docs for APO
cannot find defaults and docs for AROON
cimport numpy as np
from cython import boundscheck, wraparound
cimport _ta_lib as lib
from _ta_lib cimport TA_RetCode
# NOTE: _ta_check_success, NaN are defined in common.pxi

np.import_array() # Initialize the NumPy C API

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ACCBANDS( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ ACCBANDS(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outrealupperband
        double outrealmiddleband
        double outreallowerband
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outrealupperband = NaN
    outrealmiddleband = NaN
    outreallowerband = NaN
    retCode = lib.TA_ACCBANDS( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outrealupperband , &outrealmiddleband , &outreallowerband )
    _ta_check_success("TA_ACCBANDS", retCode)
    return outrealupperband , outrealmiddleband , outreallowerband 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ACOS( np.ndarray real not None ):
    """ ACOS(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ACOS( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ACOS", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_AD( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None ):
    """ AD(high, low, close, volume)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    volume = check_array(volume)
    volume_data = <double*>volume.data
    length = check_length4(high, low, close, volume)
    outreal = NaN
    retCode = lib.TA_AD( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , volume_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_AD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ADD( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ ADD(real0, real1)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_ADD( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ADOSC( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None , int fastperiod=-2**31 , int slowperiod=-2**31 ):
    """ ADOSC(high, low, close, volume[, fastperiod=?, slowperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    volume = check_array(volume)
    volume_data = <double*>volume.data
    length = check_length4(high, low, close, volume)
    outreal = NaN
    retCode = lib.TA_ADOSC( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , volume_data , fastperiod , slowperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ADX( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ ADX(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_ADX( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ADXR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ ADXR(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_ADXR( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADXR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_APO( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int matype=0 ):
    """ APO(real[, fastperiod=?, slowperiod=?, matype=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_APO( <int>(length) - 1 , <int>(length) - 1 , real_data , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_APO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_AROON( np.ndarray high not None , cannot find defaults and docs for AROONOSC
cannot find defaults and docs for ASIN
cannot find defaults and docs for ATAN
cannot find defaults and docs for ATR
cannot find defaults and docs for AVGPRICE
cannot find defaults and docs for AVGDEV
cannot find defaults and docs for BBANDS
cannot find defaults and docs for BETA
cannot find defaults and docs for BOP
np.ndarray low not None , int timeperiod=-2**31 ):
    """ AROON(high, low[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outaroondown
        double outaroonup
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outaroondown = NaN
    outaroonup = NaN
    retCode = lib.TA_AROON( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outaroondown , &outaroonup )
    _ta_check_success("TA_AROON", retCode)
    return outaroondown , outaroonup 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_AROONOSC( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ AROONOSC(high, low[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_AROONOSC( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_AROONOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ASIN( np.ndarray real not None ):
    """ ASIN(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ASIN( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ASIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ATAN( np.ndarray real not None ):
    """ ATAN(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ATAN( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ATAN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ATR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ ATR(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_ATR( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ATR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_AVGPRICE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ AVGPRICE(open, high, low, close)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outreal = NaN
    retCode = lib.TA_AVGPRICE( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_AVGPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_AVGDEV( np.ndarray real not None , int timeperiod=-2**31 ):
    """ AVGDEV(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_AVGDEV( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_AVGDEV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_BBANDS( np.ndarray real not None , int timeperiod=-2**31 , double nbdevup=-4e37 , double nbdevdn=-4e37 , int matype=0 ):
    """ BBANDS(real[, timeperiod=?, nbdevup=?, nbdevdn=?, matype=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outrealupperband
        double outrealmiddleband
        double outreallowerband
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outrealupperband = NaN
    outrealmiddleband = NaN
    outreallowerband = NaN
    retCode = lib.TA_BBANDS( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , nbdevup , nbdevdn , matype , &outbegidx , &outnbelement , &outrealupperband , &outrealmiddleband , &outreallowerband )
    _ta_check_success("TA_BBANDS", retCode)
    return outrealupperband , outrealmiddleband , outreallowerband 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_BETA( np.ndarray real0 not None , np.ndarray real1 not None , int timeperiod=-2**31 ):
    """ BETA(real0, real1[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_BETA( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_BETA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_BOP( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ BOP(open, high, low, close)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
cannot find defaults and docs for CCI
cannot find defaults and docs for CEIL
cannot find defaults and docs for CMO
cannot find defaults and docs for CORREL
cannot find defaults and docs for COS
cannot find defaults and docs for COSH
cannot find defaults and docs for DEMA
cannot find defaults and docs for DIV
cannot find defaults and docs for DX
cannot find defaults and docs for EMA
        double outreal
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outreal = NaN
    retCode = lib.TA_BOP( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_BOP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CCI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ CCI(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_CCI( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CCI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CEIL( np.ndarray real not None ):
    """ CEIL(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_CEIL( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CEIL", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CMO( np.ndarray real not None , int timeperiod=-2**31 ):
    """ CMO(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_CMO( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CMO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CORREL( np.ndarray real0 not None , np.ndarray real1 not None , int timeperiod=-2**31 ):
    """ CORREL(real0, real1[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_CORREL( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CORREL", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_COS( np.ndarray real not None ):
    """ COS(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_COS( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_COS", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_COSH( np.ndarray real not None ):
    """ COSH(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_COSH( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_COSH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_DEMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ DEMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_DEMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_DEMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_DIV( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ DIV(real0, real1)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_DIV( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_DIV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_DX( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ DX(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_DX( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_DX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_EMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ EMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_EMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , cannot find defaults and docs for EXP
cannot find defaults and docs for FLOOR
cannot find defaults and docs for HT_DCPERIOD
cannot find defaults and docs for HT_DCPHASE
cannot find defaults and docs for HT_PHASOR
cannot find defaults and docs for HT_SINE
cannot find defaults and docs for HT_TRENDLINE
cannot find defaults and docs for HT_TRENDMODE
cannot find defaults and docs for IMI
cannot find defaults and docs for JMA
cannot find defaults and docs for KAMA
&outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_EMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_EXP( np.ndarray real not None ):
    """ EXP(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_EXP( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_EXP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_FLOOR( np.ndarray real not None ):
    """ FLOOR(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_FLOOR( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_FLOOR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_DCPERIOD( np.ndarray real not None ):
    """ HT_DCPERIOD(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_HT_DCPERIOD( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_HT_DCPERIOD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_DCPHASE( np.ndarray real not None ):
    """ HT_DCPHASE(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_HT_DCPHASE( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_HT_DCPHASE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_PHASOR( np.ndarray real not None ):
    """ HT_PHASOR(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outinphase
        double outquadrature
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinphase = NaN
    outquadrature = NaN
    retCode = lib.TA_HT_PHASOR( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outinphase , &outquadrature )
    _ta_check_success("TA_HT_PHASOR", retCode)
    return outinphase , outquadrature 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_SINE( np.ndarray real not None ):
    """ HT_SINE(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outsine
        double outleadsine
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outsine = NaN
    outleadsine = NaN
    retCode = lib.TA_HT_SINE( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outsine , &outleadsine )
    _ta_check_success("TA_HT_SINE", retCode)
    return outsine , outleadsine 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_TRENDLINE( np.ndarray real not None ):
    """ HT_TRENDLINE(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_HT_TRENDLINE( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_HT_TRENDLINE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_TRENDMODE( np.ndarray real not None ):
    """ HT_TRENDMODE(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outinteger
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinteger = 0
    retCode = lib.TA_HT_TRENDMODE( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_HT_TRENDMODE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_IMI( np.ndarray open not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ IMI(open, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    open = check_array(open)
    open_data = <double*>open.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length2(open, close)
    outreal = NaN
    retCode = lib.TA_IMI( <int>(length) - 1 , <int>(length) - 1 , open_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_IMI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_JMA( np.ndarray real not None , int timeperiod=7 , int phase=0 , int volperiods=50 ):
    """ JMA(real[, timeperiod=?, phase=?, volperiods=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outrealjma
        double outrealupperband
        double outreallowerband
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outrealjma = NaN
    outrealupperband = NaN
    outreallowerband = NaN
    retCode = lib.TA_JMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , phase , volperiods , &outbegidx , &outnbelement , &outrealjma , &outrealupperband , &outreallowerband )
    _ta_check_success("TA_JMA", retCode)
    return outrealjma , outrealupperband , outreallowerband 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_KAMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ KAMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_KAMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_KAMA", retCode)
cannot find defaults and docs for LINEARREG
cannot find defaults and docs for LINEARREG_ANGLE
cannot find defaults and docs for LINEARREG_INTERCEPT
cannot find defaults and docs for LINEARREG_SLOPE
cannot find defaults and docs for LN
cannot find defaults and docs for LOG10
cannot find defaults and docs for MA
cannot find defaults and docs for MACD
cannot find defaults and docs for MACDEXT
cannot find defaults and docs for MACDFIX
cannot find defaults and docs for MAMA
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LINEARREG( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LINEARREG_ANGLE( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG_ANGLE(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG_ANGLE( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG_ANGLE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LINEARREG_INTERCEPT( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG_INTERCEPT(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG_INTERCEPT( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG_INTERCEPT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LINEARREG_SLOPE( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG_SLOPE(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG_SLOPE( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG_SLOPE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LN( np.ndarray real not None ):
    """ LN(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LN( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LOG10( np.ndarray real not None ):
    """ LOG10(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LOG10( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LOG10", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MA( np.ndarray real not None , int timeperiod=-2**31 , int matype=0 ):
    """ MA(real[, timeperiod=?, matype=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MACD( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int signalperiod=-2**31 ):
    """ MACD(real[, fastperiod=?, slowperiod=?, signalperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmacd
        double outmacdsignal
        double outmacdhist
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmacd = NaN
    outmacdsignal = NaN
    outmacdhist = NaN
    retCode = lib.TA_MACD( <int>(length) - 1 , <int>(length) - 1 , real_data , fastperiod , slowperiod , signalperiod , &outbegidx , &outnbelement , &outmacd , &outmacdsignal , &outmacdhist )
    _ta_check_success("TA_MACD", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MACDEXT( np.ndarray real not None , int fastperiod=-2**31 , int fastmatype=0 , int slowperiod=-2**31 , int slowmatype=0 , int signalperiod=-2**31 , int signalmatype=0 ):
    """ MACDEXT(real[, fastperiod=?, fastmatype=?, slowperiod=?, slowmatype=?, signalperiod=?, signalmatype=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmacd
        double outmacdsignal
        double outmacdhist
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmacd = NaN
    outmacdsignal = NaN
    outmacdhist = NaN
    retCode = lib.TA_MACDEXT( <int>(length) - 1 , <int>(length) - 1 , real_data , fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype , &outbegidx , &outnbelement , &outmacd , &outmacdsignal , &outmacdhist )
    _ta_check_success("TA_MACDEXT", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MACDFIX( np.ndarray real not None , int signalperiod=-2**31 ):
    """ MACDFIX(real[, signalperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmacd
        double outmacdsignal
        double outmacdhist
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmacd = NaN
    outmacdsignal = NaN
    outmacdhist = NaN
    retCode = lib.TA_MACDFIX( <int>(length) - 1 , <int>(length) - 1 , real_data , signalperiod , &outbegidx , &outnbelement , &outmacd , &outmacdsignal , &outmacdhist )
    _ta_check_success("TA_MACDFIX", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MAMA( cannot find defaults and docs for MAVP
cannot find defaults and docs for MAX
cannot find defaults and docs for MAXINDEX
cannot find defaults and docs for MEDPRICE
cannot find defaults and docs for MFI
cannot find defaults and docs for MIDPOINT
cannot find defaults and docs for MIDPRICE
cannot find defaults and docs for MIN
cannot find defaults and docs for MININDEX
cannot find defaults and docs for MINMAX
np.ndarray real not None , double fastlimit=-4e37 , double slowlimit=-4e37 ):
    """ MAMA(real[, fastlimit=?, slowlimit=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmama
        double outfama
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmama = NaN
    outfama = NaN
    retCode = lib.TA_MAMA( <int>(length) - 1 , <int>(length) - 1 , real_data , fastlimit , slowlimit , &outbegidx , &outnbelement , &outmama , &outfama )
    _ta_check_success("TA_MAMA", retCode)
    return outmama , outfama 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MAVP( np.ndarray real not None , np.ndarray periods not None , int minperiod=-2**31 , int maxperiod=-2**31 , int matype=0 ):
    """ MAVP(real, periods[, minperiod=?, maxperiod=?, matype=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        double* periods_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    periods = check_array(periods)
    periods_data = <double*>periods.data
    length = check_length2(real, periods)
    outreal = NaN
    retCode = lib.TA_MAVP( <int>(length) - 1 , <int>(length) - 1 , real_data , periods_data , minperiod , maxperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MAVP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MAX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MAX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MAX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MAX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MAXINDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MAXINDEX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outinteger
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinteger = 0
    retCode = lib.TA_MAXINDEX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_MAXINDEX", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MEDPRICE( np.ndarray high not None , np.ndarray low not None ):
    """ MEDPRICE(high, low)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_MEDPRICE( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MEDPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MFI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None , int timeperiod=-2**31 ):
    """ MFI(high, low, close, volume[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    volume = check_array(volume)
    volume_data = <double*>volume.data
    length = check_length4(high, low, close, volume)
    outreal = NaN
    retCode = lib.TA_MFI( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , volume_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MFI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MIDPOINT( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MIDPOINT(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MIDPOINT( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MIDPOINT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MIDPRICE( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ MIDPRICE(high, low[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_MIDPRICE( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MIDPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MIN( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MIN(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MIN( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MININDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MININDEX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outinteger
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinteger = 0
    retCode = lib.TA_MININDEX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_MININDEX", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MINMAX( np.ndarray real not Nonecannot find defaults and docs for MINMAXINDEX
cannot find defaults and docs for MINUS_DI
cannot find defaults and docs for MINUS_DM
cannot find defaults and docs for MOM
cannot find defaults and docs for MULT
cannot find defaults and docs for NATR
cannot find defaults and docs for OBV
cannot find defaults and docs for PLUS_DI
cannot find defaults and docs for PLUS_DM
 , int timeperiod=-2**31 ):
    """ MINMAX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmin
        double outmax
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmin = NaN
    outmax = NaN
    retCode = lib.TA_MINMAX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outmin , &outmax )
    _ta_check_success("TA_MINMAX", retCode)
    return outmin , outmax 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MINMAXINDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MINMAXINDEX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outminidx
        int outmaxidx
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outminidx = 0
    outmaxidx = 0
    retCode = lib.TA_MINMAXINDEX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outminidx , &outmaxidx )
    _ta_check_success("TA_MINMAXINDEX", retCode)
    return outminidx , outmaxidx 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MINUS_DI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ MINUS_DI(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_MINUS_DI( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MINUS_DI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MINUS_DM( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ MINUS_DM(high, low[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_MINUS_DM( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MINUS_DM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MOM( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MOM(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MOM( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MOM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MULT( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ MULT(real0, real1)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_MULT( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MULT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_NATR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ NATR(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_NATR( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_NATR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_OBV( np.ndarray real not None , np.ndarray volume not None ):
    """ OBV(real, volume)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    volume = check_array(volume)
    volume_data = <double*>volume.data
    length = check_length2(real, volume)
    outreal = NaN
    retCode = lib.TA_OBV( <int>(length) - 1 , <int>(length) - 1 , real_data , volume_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_OBV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_PLUS_DI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ PLUS_DI(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_PLUS_DI( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_PLUS_DI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_PLUS_DM( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ PLUS_DM(high, low[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
cannot find defaults and docs for PPO
cannot find defaults and docs for ROC
cannot find defaults and docs for ROCP
cannot find defaults and docs for ROCR
cannot find defaults and docs for ROCR100
cannot find defaults and docs for RSI
cannot find defaults and docs for SAR
cannot find defaults and docs for SAREXT
cannot find defaults and docs for SIN
cannot find defaults and docs for SINH
cannot find defaults and docs for SMA
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_PLUS_DM( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_PLUS_DM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_PPO( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int matype=0 ):
    """ PPO(real[, fastperiod=?, slowperiod=?, matype=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_PPO( <int>(length) - 1 , <int>(length) - 1 , real_data , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_PPO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ROC( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROC(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROC( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ROCP( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROCP(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROCP( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROCP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ROCR( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROCR(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROCR( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROCR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ROCR100( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROCR100(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROCR100( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROCR100", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_RSI( np.ndarray real not None , int timeperiod=-2**31 ):
    """ RSI(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_RSI( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_RSI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SAR( np.ndarray high not None , np.ndarray low not None , double acceleration=-4e37 , double maximum=-4e37 ):
    """ SAR(high, low[, acceleration=?, maximum=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_SAR( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , acceleration , maximum , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SAR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SAREXT( np.ndarray high not None , np.ndarray low not None , double startvalue=-4e37 , double offsetonreverse=-4e37 , double accelerationinitlong=-4e37 , double accelerationlong=-4e37 , double accelerationmaxlong=-4e37 , double accelerationinitshort=-4e37 , double accelerationshort=-4e37 , double accelerationmaxshort=-4e37 ):
    """ SAREXT(high, low[, startvalue=?, offsetonreverse=?, accelerationinitlong=?, accelerationlong=?, accelerationmaxlong=?, accelerationinitshort=?, accelerationshort=?, accelerationmaxshort=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_SAREXT( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , startvalue , offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SAREXT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SIN( np.ndarray real not None ):
    """ SIN(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SIN( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SINH( np.ndarray real not None ):
    """ SINH(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SINH( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SINH", retCode)
    return outreal 

cannot find defaults and docs for SQRT
cannot find defaults and docs for STDDEV
cannot find defaults and docs for STOCH
cannot find defaults and docs for STOCHF
cannot find defaults and docs for STOCHRSI
cannot find defaults and docs for SUB
cannot find defaults and docs for SUM
cannot find defaults and docs for T3
cannot find defaults and docs for TAN
@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ SMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SQRT( np.ndarray real not None ):
    """ SQRT(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SQRT( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SQRT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_STDDEV( np.ndarray real not None , int timeperiod=-2**31 , double nbdev=-4e37 ):
    """ STDDEV(real[, timeperiod=?, nbdev=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_STDDEV( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , nbdev , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_STDDEV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_STOCH( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int fastk_period=-2**31 , int slowk_period=-2**31 , int slowk_matype=0 , int slowd_period=-2**31 , int slowd_matype=0 ):
    """ STOCH(high, low, close[, fastk_period=?, slowk_period=?, slowk_matype=?, slowd_period=?, slowd_matype=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outslowk
        double outslowd
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outslowk = NaN
    outslowd = NaN
    retCode = lib.TA_STOCH( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype , &outbegidx , &outnbelement , &outslowk , &outslowd )
    _ta_check_success("TA_STOCH", retCode)
    return outslowk , outslowd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_STOCHF( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int fastk_period=-2**31 , int fastd_period=-2**31 , int fastd_matype=0 ):
    """ STOCHF(high, low, close[, fastk_period=?, fastd_period=?, fastd_matype=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outfastk
        double outfastd
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outfastk = NaN
    outfastd = NaN
    retCode = lib.TA_STOCHF( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , &outfastk , &outfastd )
    _ta_check_success("TA_STOCHF", retCode)
    return outfastk , outfastd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_STOCHRSI( np.ndarray real not None , int timeperiod=-2**31 , int fastk_period=-2**31 , int fastd_period=-2**31 , int fastd_matype=0 ):
    """ STOCHRSI(real[, timeperiod=?, fastk_period=?, fastd_period=?, fastd_matype=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outfastk
        double outfastd
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outfastk = NaN
    outfastd = NaN
    retCode = lib.TA_STOCHRSI( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , &outfastk , &outfastd )
    _ta_check_success("TA_STOCHRSI", retCode)
    return outfastk , outfastd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SUB( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ SUB(real0, real1)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_SUB( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SUB", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SUM( np.ndarray real not None , int timeperiod=-2**31 ):
    """ SUM(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SUM( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SUM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_T3( np.ndarray real not None , int timeperiod=-2**31 , double vfactor=-4e37 ):
    """ T3(real[, timeperiod=?, vfactor=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_T3( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , vfactor , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_T3", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TAN( np.ndarray real not None ):
    """ TAN(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
cannot find defaults and docs for TANH
cannot find defaults and docs for TEMA
cannot find defaults and docs for TRANGE
cannot find defaults and docs for TRIMA
cannot find defaults and docs for TRIX
cannot find defaults and docs for TSF
cannot find defaults and docs for TYPPRICE
cannot find defaults and docs for ULTOSC
cannot find defaults and docs for VAR
cannot find defaults and docs for WCLPRICE
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TAN( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TAN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TANH( np.ndarray real not None ):
    """ TANH(real)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TANH( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TANH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TEMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TEMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TEMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TEMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TRANGE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ TRANGE(high, low, close)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_TRANGE( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TRANGE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TRIMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TRIMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TRIMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TRIMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TRIX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TRIX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TRIX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TRIX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TSF( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TSF(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TSF( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TSF", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TYPPRICE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ TYPPRICE(high, low, close)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_TYPPRICE( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TYPPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ULTOSC( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod1=-2**31 , int timeperiod2=-2**31 , int timeperiod3=-2**31 ):
    """ ULTOSC(high, low, close[, timeperiod1=?, timeperiod2=?, timeperiod3=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_ULTOSC( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod1 , timeperiod2 , timeperiod3 , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ULTOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_VAR( np.ndarray real not None , int timeperiod=-2**31 , double nbdev=-4e37 ):
    """ VAR(real[, timeperiod=?, nbdev=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_VAR( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , nbdev , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_VAR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_WCLPRICE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ WCLPRICE(high, low, close)"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
cannot find defaults and docs for WILLR
cannot find defaults and docs for WMA
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_WCLPRICE( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_WCLPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_WILLR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ WILLR(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_WILLR( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_WILLR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_WMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ WMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_WMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_WMA", retCode)
    return outreal 

