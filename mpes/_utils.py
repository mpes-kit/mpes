#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""
import numpy as np

def numFormatConversion(seq, form='int', **kwds):
    """
    When length keyword is not specified as an argument, the function
    returns a format-converted sequence of numbers
    
    The function returns nothing when the conversion fails due to errors
    
    **Parameters**
    
    seq : 1D numeric array
        the numeric array to be converted
    form : str | 'int'
        the converted format
    
    **Return**
    
    numseq : converted numeric type
        the format-converted array
    """
    
    try:
        lseq = len(seq)
    except:
        raise
    
    l = kwds.pop('length', lseq)
    if lseq == l:
        # Case of numeric array of the right length but may not be
        # the right type
        try:
            numseq = map(eval(form), seq)
            return numseq
        except:
            raise 
    else:
        # Case of numeric array of the right type but wrong length
        return
        

def revaxis(arr, axis=-1):
    """
    Reverse an ndarray along certain axis
    
    **Parameters**
    arr : nD numeric array
        array to invert
    axis : int | -1
        the axis along which to invert
    
    **Return**
    revarr : nD numeric array
        axis-inverted nD array
    """
    
    arr = np.asarray(arr).swapaxes(axis, 0)
    arr = arr[::-1,...]
    revarr = arr.swapaxes(0, axis)
    return revarr