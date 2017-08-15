#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from math import cos, pi
import numpy as np

def to_odd(num):
    """
    Convert to nearest odd number
	
	**Parameters**
	
	num : float/int
	
	**Return**
	
	oddnum : int
		the nearest odd number
    """

    rem = np.round(num) % 2
	oddnum = num + int(cos(rem*pi/2))
	
    return oddnum
	

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