cannot find defaults and docs for ACCBANDS
cannot find defaults and docs for ACOS
cannot find defaults and docs for AD
cannot find defaults and docs for ADD
cimport numpy as np
from numpy import nan
from cython import boundscheck, wraparound

# _ta_check_success: defined in _common.pxi

cdef double NaN = nan

cdef extern from "numpy/arrayobject.h":
    int PyArray_TYPE(np.ndarray)
    np.ndarray PyArray_EMPTY(int, np.npy_intp*, int, int)
    int PyArray_FLAGS(np.ndarray)
    np.ndarray PyArray_GETCONTIGUOUS(np.ndarray)

np.import_array() # Initialize the NumPy C API

cimport _ta_lib as lib
from _ta_lib cimport TA_RetCode

cdef np.ndarray check_array(np.ndarray real):
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("input array type is not double")
    if real.ndim != 1:
        raise Exception("input array has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_ARRAY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    return real

cdef np.npy_intp check_length2(np.ndarray a1, np.ndarray a2) except -1:
    cdef:
        np.npy_intp length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    return length

cdef np.npy_intp check_length3(np.ndarray a1, np.ndarray a2, np.ndarray a3) except -1:
    cdef:
        np.npy_intp length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    if length != a3.shape[0]:
        raise Exception("input array lengths are different")
    return length

cdef np.npy_intp check_length4(np.ndarray a1, np.ndarray a2, np.ndarray a3, np.ndarray a4) except -1:
    cdef:
        np.npy_intp length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    if length != a3.shape[0]:
        raise Exception("input array lengths are different")
    if length != a4.shape[0]:
        raise Exception("input array lengths are different")
    return length

cdef np.npy_int check_begidx1(np.npy_intp length, double* a1):
    cdef:
        double val
    for i from 0 <= i < length:
        val = a1[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.npy_int check_begidx2(np.npy_intp length, double* a1, double* a2):
    cdef:
        double val
    for i from 0 <= i < length:
        val = a1[i]
        if val != val:
            continue
        val = a2[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.npy_int check_begidx3(np.npy_intp length, double* a1, double* a2, double* a3):
    cdef:
        double val
    for i from 0 <= i < length:
        val = a1[i]
        if val != val:
            continue
        val = a2[i]
        if val != val:
            continue
        val = a3[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.npy_int check_begidx4(np.npy_intp length, double* a1, double* a2, double* a3, double* a4):
    cdef:
        double val
    for i from 0 <= i < length:
        val = a1[i]
        if val != val:
            continue
        val = a2[i]
        if val != val:
            continue
        val = a3[i]
        if val != val:
            continue
        val = a4[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.ndarray make_double_array(np.npy_intp length, int lookback):
    cdef:
        np.ndarray outreal
        double* outreal_data
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_ARRAY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    return outreal

cdef np.ndarray make_int_array(np.npy_intp length, int lookback):
    cdef:
        np.ndarray outinteger
        int* outinteger_data
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_ARRAY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    return outinteger


@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ACCBANDS( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ ACCBANDS(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outrealupperband
        np.ndarray outrealmiddleband
        np.ndarray outreallowerband
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ACCBANDS_Lookback( timeperiod )
    outrealupperband = make_double_array(length, lookback)
    outrealmiddleband = make_double_array(length, lookback)
    outreallowerband = make_double_array(length, lookback)
    retCode = lib.TA_ACCBANDS( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outrealupperband.data)+lookback , <double *>(outrealmiddleband.data)+lookback , <double *>(outreallowerband.data)+lookback )
    _ta_check_success("TA_ACCBANDS", retCode)
    return outrealupperband , outrealmiddleband , outreallowerband 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ACOS( np.ndarray real not None ):
    """ ACOS(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ACOS_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ACOS( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ACOS", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AD( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None ):
    """ AD(high, low, close, volume)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    volume = check_array(volume)
    length = check_length4(high, low, close, volume)
    begidx = check_begidx4(length, <double*>(high.data), <double*>(low.data), <double*>(close.data), <double*>(volume.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_AD_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_AD( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , <double *>(volume.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_AD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADD( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ ADD(real0, real1)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
cannot find defaults and docs for ADOSC
cannot find defaults and docs for ADX
cannot find defaults and docs for ADXR
cannot find defaults and docs for APO
cannot find defaults and docs for AROON
cannot find defaults and docs for AROONOSC
cannot find defaults and docs for ASIN
cannot find defaults and docs for ATAN
    lookback = begidx + lib.TA_ADD_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ADD( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ADD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADOSC( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None , int fastperiod=-2**31 , int slowperiod=-2**31 ):
    """ ADOSC(high, low, close, volume[, fastperiod=?, slowperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    volume = check_array(volume)
    length = check_length4(high, low, close, volume)
    begidx = check_begidx4(length, <double*>(high.data), <double*>(low.data), <double*>(close.data), <double*>(volume.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ADOSC_Lookback( fastperiod , slowperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ADOSC( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , <double *>(volume.data)+begidx , fastperiod , slowperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ADOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADX( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ ADX(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ADX_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ADX( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ADX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADXR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ ADXR(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ADXR_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ADXR( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ADXR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def APO( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int matype=0 ):
    """ APO(real[, fastperiod=?, slowperiod=?, matype=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_APO_Lookback( fastperiod , slowperiod , matype )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_APO( 0 , endidx , <double *>(real.data)+begidx , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_APO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AROON( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ AROON(high, low[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outaroondown
        np.ndarray outaroonup
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_AROON_Lookback( timeperiod )
    outaroondown = make_double_array(length, lookback)
    outaroonup = make_double_array(length, lookback)
    retCode = lib.TA_AROON( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outaroondown.data)+lookback , <double *>(outaroonup.data)+lookback )
    _ta_check_success("TA_AROON", retCode)
    return outaroondown , outaroonup 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AROONOSC( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ AROONOSC(high, low[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_AROONOSC_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_AROONOSC( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_AROONOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ASIN( np.ndarray real not None ):
    """ ASIN(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ASIN_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ASIN( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ASIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ATAN( np.ndarray real not None ):
    """ ATAN(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
cannot find defaults and docs for ATR
cannot find defaults and docs for AVGPRICE
cannot find defaults and docs for AVGDEV
cannot find defaults and docs for BBANDS
cannot find defaults and docs for BETA
cannot find defaults and docs for BOP
cannot find defaults and docs for CCI
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ATAN_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ATAN( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ATAN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ATR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ ATR(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ATR_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ATR( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ATR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AVGPRICE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ AVGPRICE(open, high, low, close)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_AVGPRICE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_AVGPRICE( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_AVGPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AVGDEV( np.ndarray real not None , int timeperiod=-2**31 ):
    """ AVGDEV(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_AVGDEV_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_AVGDEV( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_AVGDEV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def BBANDS( np.ndarray real not None , int timeperiod=-2**31 , double nbdevup=-4e37 , double nbdevdn=-4e37 , int matype=0 ):
    """ BBANDS(real[, timeperiod=?, nbdevup=?, nbdevdn=?, matype=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outrealupperband
        np.ndarray outrealmiddleband
        np.ndarray outreallowerband
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_BBANDS_Lookback( timeperiod , nbdevup , nbdevdn , matype )
    outrealupperband = make_double_array(length, lookback)
    outrealmiddleband = make_double_array(length, lookback)
    outreallowerband = make_double_array(length, lookback)
    retCode = lib.TA_BBANDS( 0 , endidx , <double *>(real.data)+begidx , timeperiod , nbdevup , nbdevdn , matype , &outbegidx , &outnbelement , <double *>(outrealupperband.data)+lookback , <double *>(outrealmiddleband.data)+lookback , <double *>(outreallowerband.data)+lookback )
    _ta_check_success("TA_BBANDS", retCode)
    return outrealupperband , outrealmiddleband , outreallowerband 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def BETA( np.ndarray real0 not None , np.ndarray real1 not None , int timeperiod=-2**31 ):
    """ BETA(real0, real1[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_BETA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_BETA( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_BETA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def BOP( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ BOP(open, high, low, close)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_BOP_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_BOP( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_BOP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CCI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ CCI(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CCI_Lookback( timeperiod )
cannot find defaults and docs for CEIL
cannot find defaults and docs for CMO
cannot find defaults and docs for CORREL
cannot find defaults and docs for COS
cannot find defaults and docs for COSH
cannot find defaults and docs for DEMA
cannot find defaults and docs for DIV
cannot find defaults and docs for DX
cannot find defaults and docs for EMA
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_CCI( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_CCI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CEIL( np.ndarray real not None ):
    """ CEIL(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CEIL_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_CEIL( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_CEIL", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CMO( np.ndarray real not None , int timeperiod=-2**31 ):
    """ CMO(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CMO_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_CMO( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_CMO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CORREL( np.ndarray real0 not None , np.ndarray real1 not None , int timeperiod=-2**31 ):
    """ CORREL(real0, real1[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CORREL_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_CORREL( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_CORREL", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def COS( np.ndarray real not None ):
    """ COS(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_COS_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_COS( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_COS", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def COSH( np.ndarray real not None ):
    """ COSH(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_COSH_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_COSH( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_COSH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def DEMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ DEMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_DEMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_DEMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_DEMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def DIV( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ DIV(real0, real1)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_DIV_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_DIV( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_DIV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def DX( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ DX(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_DX_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_DX( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_DX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def EMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ EMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
cannot find defaults and docs for EXP
cannot find defaults and docs for FLOOR
cannot find defaults and docs for HT_DCPERIOD
cannot find defaults and docs for HT_DCPHASE
cannot find defaults and docs for HT_PHASOR
cannot find defaults and docs for HT_SINE
cannot find defaults and docs for HT_TRENDLINE
cannot find defaults and docs for HT_TRENDMODE
cannot find defaults and docs for IMI
    lookback = begidx + lib.TA_EMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_EMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_EMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def EXP( np.ndarray real not None ):
    """ EXP(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_EXP_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_EXP( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_EXP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def FLOOR( np.ndarray real not None ):
    """ FLOOR(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_FLOOR_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_FLOOR( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_FLOOR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_DCPERIOD( np.ndarray real not None ):
    """ HT_DCPERIOD(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_DCPERIOD_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_HT_DCPERIOD( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_HT_DCPERIOD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_DCPHASE( np.ndarray real not None ):
    """ HT_DCPHASE(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_DCPHASE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_HT_DCPHASE( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_HT_DCPHASE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_PHASOR( np.ndarray real not None ):
    """ HT_PHASOR(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinphase
        np.ndarray outquadrature
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_PHASOR_Lookback( )
    outinphase = make_double_array(length, lookback)
    outquadrature = make_double_array(length, lookback)
    retCode = lib.TA_HT_PHASOR( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outinphase.data)+lookback , <double *>(outquadrature.data)+lookback )
    _ta_check_success("TA_HT_PHASOR", retCode)
    return outinphase , outquadrature 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_SINE( np.ndarray real not None ):
    """ HT_SINE(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outsine
        np.ndarray outleadsine
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_SINE_Lookback( )
    outsine = make_double_array(length, lookback)
    outleadsine = make_double_array(length, lookback)
    retCode = lib.TA_HT_SINE( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outsine.data)+lookback , <double *>(outleadsine.data)+lookback )
    _ta_check_success("TA_HT_SINE", retCode)
    return outsine , outleadsine 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_TRENDLINE( np.ndarray real not None ):
    """ HT_TRENDLINE(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_TRENDLINE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_HT_TRENDLINE( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_HT_TRENDLINE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_TRENDMODE( np.ndarray real not None ):
    """ HT_TRENDMODE(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_TRENDMODE_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_HT_TRENDMODE( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_HT_TRENDMODE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def IMI( np.ndarray open not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ IMI(open, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    open = check_array(open)
    close = check_array(close)
    length = check_length2(open, close)
    begidx = check_begidx2(length, <double*>(open.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_IMI_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_IMI( 0 , endidx , <double *>(open.data)+begidx , cannot find defaults and docs for JMA
cannot find defaults and docs for KAMA
cannot find defaults and docs for LINEARREG
cannot find defaults and docs for LINEARREG_ANGLE
cannot find defaults and docs for LINEARREG_INTERCEPT
cannot find defaults and docs for LINEARREG_SLOPE
cannot find defaults and docs for LN
cannot find defaults and docs for LOG10
cannot find defaults and docs for MA
<double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_IMI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def JMA( np.ndarray real not None , int timeperiod=7 , int phase=0 , int volperiods=50 ):
    """ JMA(real[, timeperiod=?, phase=?, volperiods=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outrealjma
        np.ndarray outrealupperband
        np.ndarray outreallowerband
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_JMA_Lookback( timeperiod , phase , volperiods )
    outrealjma = make_double_array(length, lookback)
    outrealupperband = make_double_array(length, lookback)
    outreallowerband = make_double_array(length, lookback)
    retCode = lib.TA_JMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , phase , volperiods , &outbegidx , &outnbelement , <double *>(outrealjma.data)+lookback , <double *>(outrealupperband.data)+lookback , <double *>(outreallowerband.data)+lookback )
    _ta_check_success("TA_JMA", retCode)
    return outrealjma , outrealupperband , outreallowerband 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def KAMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ KAMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_KAMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_KAMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_KAMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LINEARREG_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LINEARREG( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LINEARREG", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG_ANGLE( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG_ANGLE(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LINEARREG_ANGLE_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LINEARREG_ANGLE( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LINEARREG_ANGLE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG_INTERCEPT( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG_INTERCEPT(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LINEARREG_INTERCEPT_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LINEARREG_INTERCEPT( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LINEARREG_INTERCEPT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG_SLOPE( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG_SLOPE(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LINEARREG_SLOPE_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LINEARREG_SLOPE( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LINEARREG_SLOPE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LN( np.ndarray real not None ):
    """ LN(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LN_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LN( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LOG10( np.ndarray real not None ):
    """ LOG10(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LOG10_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LOG10( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LOG10", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MA( np.ndarray real not None , int timeperiod=-2**31 , int matype=0 ):
    """ MA(real[, timeperiod=?, matype=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
cannot find defaults and docs for MACD
cannot find defaults and docs for MACDEXT
cannot find defaults and docs for MACDFIX
cannot find defaults and docs for MAMA
cannot find defaults and docs for MAVP
cannot find defaults and docs for MAX
cannot find defaults and docs for MAXINDEX
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MA_Lookback( timeperiod , matype )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , matype , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MACD( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int signalperiod=-2**31 ):
    """ MACD(real[, fastperiod=?, slowperiod=?, signalperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outmacd
        np.ndarray outmacdsignal
        np.ndarray outmacdhist
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MACD_Lookback( fastperiod , slowperiod , signalperiod )
    outmacd = make_double_array(length, lookback)
    outmacdsignal = make_double_array(length, lookback)
    outmacdhist = make_double_array(length, lookback)
    retCode = lib.TA_MACD( 0 , endidx , <double *>(real.data)+begidx , fastperiod , slowperiod , signalperiod , &outbegidx , &outnbelement , <double *>(outmacd.data)+lookback , <double *>(outmacdsignal.data)+lookback , <double *>(outmacdhist.data)+lookback )
    _ta_check_success("TA_MACD", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MACDEXT( np.ndarray real not None , int fastperiod=-2**31 , int fastmatype=0 , int slowperiod=-2**31 , int slowmatype=0 , int signalperiod=-2**31 , int signalmatype=0 ):
    """ MACDEXT(real[, fastperiod=?, fastmatype=?, slowperiod=?, slowmatype=?, signalperiod=?, signalmatype=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outmacd
        np.ndarray outmacdsignal
        np.ndarray outmacdhist
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MACDEXT_Lookback( fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype )
    outmacd = make_double_array(length, lookback)
    outmacdsignal = make_double_array(length, lookback)
    outmacdhist = make_double_array(length, lookback)
    retCode = lib.TA_MACDEXT( 0 , endidx , <double *>(real.data)+begidx , fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype , &outbegidx , &outnbelement , <double *>(outmacd.data)+lookback , <double *>(outmacdsignal.data)+lookback , <double *>(outmacdhist.data)+lookback )
    _ta_check_success("TA_MACDEXT", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MACDFIX( np.ndarray real not None , int signalperiod=-2**31 ):
    """ MACDFIX(real[, signalperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outmacd
        np.ndarray outmacdsignal
        np.ndarray outmacdhist
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MACDFIX_Lookback( signalperiod )
    outmacd = make_double_array(length, lookback)
    outmacdsignal = make_double_array(length, lookback)
    outmacdhist = make_double_array(length, lookback)
    retCode = lib.TA_MACDFIX( 0 , endidx , <double *>(real.data)+begidx , signalperiod , &outbegidx , &outnbelement , <double *>(outmacd.data)+lookback , <double *>(outmacdsignal.data)+lookback , <double *>(outmacdhist.data)+lookback )
    _ta_check_success("TA_MACDFIX", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAMA( np.ndarray real not None , double fastlimit=-4e37 , double slowlimit=-4e37 ):
    """ MAMA(real[, fastlimit=?, slowlimit=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outmama
        np.ndarray outfama
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MAMA_Lookback( fastlimit , slowlimit )
    outmama = make_double_array(length, lookback)
    outfama = make_double_array(length, lookback)
    retCode = lib.TA_MAMA( 0 , endidx , <double *>(real.data)+begidx , fastlimit , slowlimit , &outbegidx , &outnbelement , <double *>(outmama.data)+lookback , <double *>(outfama.data)+lookback )
    _ta_check_success("TA_MAMA", retCode)
    return outmama , outfama 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAVP( np.ndarray real not None , np.ndarray periods not None , int minperiod=-2**31 , int maxperiod=-2**31 , int matype=0 ):
    """ MAVP(real, periods[, minperiod=?, maxperiod=?, matype=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    periods = check_array(periods)
    length = check_length2(real, periods)
    begidx = check_begidx2(length, <double*>(real.data), <double*>(periods.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MAVP_Lookback( minperiod , maxperiod , matype )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MAVP( 0 , endidx , <double *>(real.data)+begidx , <double *>(periods.data)+begidx , minperiod , maxperiod , matype , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MAVP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MAX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MAX_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MAX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MAX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAXINDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MAXINDEX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MAXINDEX_Lookback( timeperiod )
cannot find defaults and docs for MEDPRICE
cannot find defaults and docs for MFI
cannot find defaults and docs for MIDPOINT
cannot find defaults and docs for MIDPRICE
cannot find defaults and docs for MIN
cannot find defaults and docs for MININDEX
cannot find defaults and docs for MINMAX
cannot find defaults and docs for MINMAXINDEX
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_MAXINDEX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_MAXINDEX", retCode)
    outinteger_data = <int*>outinteger.data
    for i from lookback <= i < length:
        outinteger_data[i] += begidx
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MEDPRICE( np.ndarray high not None , np.ndarray low not None ):
    """ MEDPRICE(high, low)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MEDPRICE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MEDPRICE( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MEDPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MFI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None , int timeperiod=-2**31 ):
    """ MFI(high, low, close, volume[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    volume = check_array(volume)
    length = check_length4(high, low, close, volume)
    begidx = check_begidx4(length, <double*>(high.data), <double*>(low.data), <double*>(close.data), <double*>(volume.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MFI_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MFI( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , <double *>(volume.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MFI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MIDPOINT( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MIDPOINT(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MIDPOINT_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MIDPOINT( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MIDPOINT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MIDPRICE( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ MIDPRICE(high, low[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MIDPRICE_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MIDPRICE( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MIDPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MIN( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MIN(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MIN_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MIN( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MININDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MININDEX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MININDEX_Lookback( timeperiod )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_MININDEX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_MININDEX", retCode)
    outinteger_data = <int*>outinteger.data
    for i from lookback <= i < length:
        outinteger_data[i] += begidx
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINMAX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MINMAX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outmin
        np.ndarray outmax
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MINMAX_Lookback( timeperiod )
    outmin = make_double_array(length, lookback)
    outmax = make_double_array(length, lookback)
    retCode = lib.TA_MINMAX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outmin.data)+lookback , <double *>(outmax.data)+lookback )
    _ta_check_success("TA_MINMAX", retCode)
    return outmin , outmax 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINMAXINDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MINMAXINDEX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outminidx
        np.ndarray outmaxidx
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MINMAXINDEX_Lookback( timeperiod )
    outminidx = make_int_array(length, lookback)
cannot find defaults and docs for MINUS_DI
cannot find defaults and docs for MINUS_DM
cannot find defaults and docs for MOM
cannot find defaults and docs for MULT
cannot find defaults and docs for NATR
cannot find defaults and docs for OBV
cannot find defaults and docs for PLUS_DI
cannot find defaults and docs for PLUS_DM
    outmaxidx = make_int_array(length, lookback)
    retCode = lib.TA_MINMAXINDEX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <int *>(outminidx.data)+lookback , <int *>(outmaxidx.data)+lookback )
    _ta_check_success("TA_MINMAXINDEX", retCode)
    outminidx_data = <int*>outminidx.data
    for i from lookback <= i < length:
        outminidx_data[i] += begidx
    outmaxidx_data = <int*>outmaxidx.data
    for i from lookback <= i < length:
        outmaxidx_data[i] += begidx
    return outminidx , outmaxidx 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINUS_DI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ MINUS_DI(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MINUS_DI_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MINUS_DI( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MINUS_DI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINUS_DM( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ MINUS_DM(high, low[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MINUS_DM_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MINUS_DM( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MINUS_DM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MOM( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MOM(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MOM_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MOM( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MOM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MULT( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ MULT(real0, real1)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MULT_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MULT( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MULT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def NATR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ NATR(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_NATR_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_NATR( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_NATR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def OBV( np.ndarray real not None , np.ndarray volume not None ):
    """ OBV(real, volume)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    volume = check_array(volume)
    length = check_length2(real, volume)
    begidx = check_begidx2(length, <double*>(real.data), <double*>(volume.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_OBV_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_OBV( 0 , endidx , <double *>(real.data)+begidx , <double *>(volume.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_OBV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def PLUS_DI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ PLUS_DI(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_PLUS_DI_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_PLUS_DI( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_PLUS_DI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def PLUS_DM( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ PLUS_DM(high, low[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
cannot find defaults and docs for PPO
cannot find defaults and docs for ROC
cannot find defaults and docs for ROCP
cannot find defaults and docs for ROCR
cannot find defaults and docs for ROCR100
cannot find defaults and docs for RSI
cannot find defaults and docs for SAR
cannot find defaults and docs for SAREXT
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_PLUS_DM_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_PLUS_DM( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_PLUS_DM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def PPO( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int matype=0 ):
    """ PPO(real[, fastperiod=?, slowperiod=?, matype=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_PPO_Lookback( fastperiod , slowperiod , matype )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_PPO( 0 , endidx , <double *>(real.data)+begidx , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_PPO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROC( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROC(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ROC_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ROC( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ROC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROCP( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROCP(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ROCP_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ROCP( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ROCP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROCR( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROCR(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ROCR_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ROCR( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ROCR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROCR100( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROCR100(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ROCR100_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ROCR100( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ROCR100", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def RSI( np.ndarray real not None , int timeperiod=-2**31 ):
    """ RSI(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_RSI_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_RSI( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_RSI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SAR( np.ndarray high not None , np.ndarray low not None , double acceleration=-4e37 , double maximum=-4e37 ):
    """ SAR(high, low[, acceleration=?, maximum=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SAR_Lookback( acceleration , maximum )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SAR( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , acceleration , maximum , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SAR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SAREXT( np.ndarray high not None , np.ndarray low not None , double startvalue=-4e37 , double offsetonreverse=-4e37 , double accelerationinitlong=-4e37 , double accelerationlong=-4e37 , double accelerationmaxlong=-4e37 , double accelerationinitshort=-4e37 , double accelerationshort=-4e37 , double accelerationmaxshort=-4e37 ):
    """ SAREXT(high, low[, startvalue=?, offsetonreverse=?, accelerationinitlong=?, accelerationlong=?, accelerationmaxlong=?, accelerationinitshort=?, accelerationshort=?, accelerationmaxshort=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SAREXT_Lookback( startvalue , cannot find defaults and docs for SIN
cannot find defaults and docs for SINH
cannot find defaults and docs for SMA
cannot find defaults and docs for SQRT
cannot find defaults and docs for STDDEV
cannot find defaults and docs for STOCH
cannot find defaults and docs for STOCHF
cannot find defaults and docs for STOCHRSI
offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SAREXT( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , startvalue , offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SAREXT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SIN( np.ndarray real not None ):
    """ SIN(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SIN_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SIN( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SINH( np.ndarray real not None ):
    """ SINH(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SINH_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SINH( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SINH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ SMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SQRT( np.ndarray real not None ):
    """ SQRT(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SQRT_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SQRT( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SQRT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STDDEV( np.ndarray real not None , int timeperiod=-2**31 , double nbdev=-4e37 ):
    """ STDDEV(real[, timeperiod=?, nbdev=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_STDDEV_Lookback( timeperiod , nbdev )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_STDDEV( 0 , endidx , <double *>(real.data)+begidx , timeperiod , nbdev , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_STDDEV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STOCH( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int fastk_period=-2**31 , int slowk_period=-2**31 , int slowk_matype=0 , int slowd_period=-2**31 , int slowd_matype=0 ):
    """ STOCH(high, low, close[, fastk_period=?, slowk_period=?, slowk_matype=?, slowd_period=?, slowd_matype=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outslowk
        np.ndarray outslowd
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_STOCH_Lookback( fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype )
    outslowk = make_double_array(length, lookback)
    outslowd = make_double_array(length, lookback)
    retCode = lib.TA_STOCH( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype , &outbegidx , &outnbelement , <double *>(outslowk.data)+lookback , <double *>(outslowd.data)+lookback )
    _ta_check_success("TA_STOCH", retCode)
    return outslowk , outslowd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STOCHF( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int fastk_period=-2**31 , int fastd_period=-2**31 , int fastd_matype=0 ):
    """ STOCHF(high, low, close[, fastk_period=?, fastd_period=?, fastd_matype=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outfastk
        np.ndarray outfastd
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_STOCHF_Lookback( fastk_period , fastd_period , fastd_matype )
    outfastk = make_double_array(length, lookback)
    outfastd = make_double_array(length, lookback)
    retCode = lib.TA_STOCHF( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , <double *>(outfastk.data)+lookback , <double *>(outfastd.data)+lookback )
    _ta_check_success("TA_STOCHF", retCode)
    return outfastk , outfastd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STOCHRSI( np.ndarray real not None , int timeperiod=-2**31 , int fastk_period=-2**31 , int fastd_period=-2**31 , int fastd_matype=0 ):
    """ STOCHRSI(real[, timeperiod=?, fastk_period=?, fastd_period=?, fastd_matype=?])"""
cannot find defaults and docs for SUB
cannot find defaults and docs for SUM
cannot find defaults and docs for T3
cannot find defaults and docs for TAN
cannot find defaults and docs for TANH
cannot find defaults and docs for TEMA
cannot find defaults and docs for TRANGE
cannot find defaults and docs for TRIMA
cannot find defaults and docs for TRIX
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outfastk
        np.ndarray outfastd
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_STOCHRSI_Lookback( timeperiod , fastk_period , fastd_period , fastd_matype )
    outfastk = make_double_array(length, lookback)
    outfastd = make_double_array(length, lookback)
    retCode = lib.TA_STOCHRSI( 0 , endidx , <double *>(real.data)+begidx , timeperiod , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , <double *>(outfastk.data)+lookback , <double *>(outfastd.data)+lookback )
    _ta_check_success("TA_STOCHRSI", retCode)
    return outfastk , outfastd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SUB( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ SUB(real0, real1)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SUB_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SUB( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SUB", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SUM( np.ndarray real not None , int timeperiod=-2**31 ):
    """ SUM(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SUM_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SUM( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SUM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def T3( np.ndarray real not None , int timeperiod=-2**31 , double vfactor=-4e37 ):
    """ T3(real[, timeperiod=?, vfactor=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_T3_Lookback( timeperiod , vfactor )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_T3( 0 , endidx , <double *>(real.data)+begidx , timeperiod , vfactor , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_T3", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TAN( np.ndarray real not None ):
    """ TAN(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TAN_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TAN( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TAN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TANH( np.ndarray real not None ):
    """ TANH(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TANH_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TANH( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TANH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TEMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TEMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TEMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TEMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TEMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TRANGE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ TRANGE(high, low, close)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TRANGE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TRANGE( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TRANGE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TRIMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TRIMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TRIMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TRIMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TRIMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
cannot find defaults and docs for TSF
cannot find defaults and docs for TYPPRICE
cannot find defaults and docs for ULTOSC
cannot find defaults and docs for VAR
cannot find defaults and docs for WCLPRICE
cannot find defaults and docs for WILLR
cannot find defaults and docs for WMA
@boundscheck(False) # turn off bounds-checking for entire function
def TRIX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TRIX(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TRIX_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TRIX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TRIX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TSF( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TSF(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TSF_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TSF( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TSF", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TYPPRICE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ TYPPRICE(high, low, close)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TYPPRICE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TYPPRICE( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TYPPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ULTOSC( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod1=-2**31 , int timeperiod2=-2**31 , int timeperiod3=-2**31 ):
    """ ULTOSC(high, low, close[, timeperiod1=?, timeperiod2=?, timeperiod3=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ULTOSC_Lookback( timeperiod1 , timeperiod2 , timeperiod3 )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ULTOSC( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod1 , timeperiod2 , timeperiod3 , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ULTOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def VAR( np.ndarray real not None , int timeperiod=-2**31 , double nbdev=-4e37 ):
    """ VAR(real[, timeperiod=?, nbdev=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_VAR_Lookback( timeperiod , nbdev )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_VAR( 0 , endidx , <double *>(real.data)+begidx , timeperiod , nbdev , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_VAR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def WCLPRICE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ WCLPRICE(high, low, close)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_WCLPRICE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_WCLPRICE( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_WCLPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def WILLR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ WILLR(high, low, close[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_WILLR_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_WILLR( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_WILLR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def WMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ WMA(real[, timeperiod=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_WMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_WMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_WMA", retCode)
    return outreal 

__TA_FUNCTION_NAMES__ = ["ACCBANDS","ACOS","AD","ADD","ADOSC","ADX","ADXR","APO","AROON","AROONOSC","ASIN","ATAN","ATR","AVGPRICE","AVGDEV","BBANDS","BETA","BOP","CCI","CEIL","CMO","CORREL","COS","COSH","DEMA","DIV","DX","EMA","EXP","FLOOR","HT_DCPERIOD","HT_DCPHASE","HT_PHASOR","HT_SINE","HT_TRENDLINE","HT_TRENDMODE","IMI","JMA","KAMA","LINEARREG","LINEARREG_ANGLE","LINEARREG_INTERCEPT","LINEARREG_SLOPE","LN","LOG10","MA","MACD","MACDEXT","MACDFIX","MAMA","MAVP","MAX","MAXINDEX","MEDPRICE","MFI","MIDPOINT","MIDPRICE","MIN","MININDEX","MINMAX","MINMAXINDEX","MINUS_DI","MINUS_DM","MOM","MULT","NATR","OBV","PLUS_DI","PLUS_DM","PPO","ROC","ROCP","ROCR","ROCR100","RSI","SAR","SAREXT","SIN","SINH","SMA","SQRT","STDDEV","STOCH","STOCHF","STOCHRSI","SUB","SUM","T3","TAN","TANH","TEMA","TRANGE","TRIMA","TRIX","TSF","TYPPRICE","ULTOSC","VAR","WCLPRICE","WILLR","WMA"]
