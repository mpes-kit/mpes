#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from __future__ import print_function, division
from math import cos, pi
from skimage import measure, filters
import numpy as np


def segment2d(img, nbands=1, **kwds):
    """
    Electronic band segmentation using local thresholding
    and connected component labeling

    **Parameters**
    
    img : numeric 2D array
        the 2D matrix to segment
    nbands : int
        number of electronic bands
    **kwds : keyword arguments

    **Return**
    
    imglabeled : labeled mask
    """

    nlabel  =  0
    dmax = to_odd(max(img.shape))
    i = 0
    blocksize = dmax - 2 * i

    while (nlabel != nbands) or (blocksize <= 0):

        binadpt = filters.threshold_local(
    img, blocksize, method='gaussian', offset=10, mode='reflect')
        imglabeled, nlabel = measure.label(img > binadpt, return_num=True)
        i += 1
        blocksize = dmax - 2 * i

    return imglabeled


def to_odd(num):
    """
    Convert to nearest odd number
    """

    rem = np.round(num) % 2
    return num + int(cos(rem*pi/2))
