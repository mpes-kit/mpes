#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""
from math import cos, pi
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
            numseq = eval('list(map(' + form + ', seq))')
            return numseq
        except:
            raise
    else:
        # Case of numeric array of the right type but wrong length
        return seq


def to_odd(num):
    """
    Convert a single number to its nearest odd number

    **Parameters**

    num : float/int

    **Return**

    oddnum : int
        the nearest odd number
    """

    rem = round(num) % 2
    oddnum = num + int(cos(rem*pi/2))

    return oddnum


def intify(*nums):
    """ Safely convert to integer (avoiding None)
    """

    intnums = list(nums) # Make a copy of the to-be-converted list
    for i, num in enumerate(nums):
        try:
            intnums[i] = int(num)
        except TypeError:
            pass

    return intnums


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


def replist(entry, row, column):
    """
    Generator of nested lists with identical entries.
    Generated values are independent of one another.

    ***Parameters***

    entry : numeric/str
        repeated item in nested list
    row : int
        number of rows in nested list
    column : int
        number of columns in nested list

    ***Return***

    nested list
    """

    return [[entry]*column for _ in range(row)]


def shuffleaxis(arr, axes, direction='front'):
    """
    Move multiple axes of a multidimensional array simultaneously
    to the front or end of its axis order

    ***Parameters***

    arr : ndarray
        array to be shuffled
    axes : tuple of int
        dimensions to be shuffled
    direction : str | 'front'
        direction of shuffling ('front' or 'end')

    ***Return***

    sharr : ndarray
        dimension-shuffled array
    """

    nax, maxaxes, minaxes = len(axes), max(axes), min(axes)
    ndim = np.ndim(arr)

    if nax > ndim:
        raise Exception('Input array has fewer dimensions than specified axes!')
    elif maxaxes > ndim-1 or minaxes < -ndim:
        raise Exception("At least one of the input axes doesn't exist!")
    else:
        if direction == 'front':
            shuffled_order = list(range(len(axes)))
        elif direction == 'end':
            shuffled_order = list(range(-len(axes),0))

    sharr = np.moveaxis(arr, axes, shuffled_order)

    return sharr


def dictmerge(D, others):
    """ Merge a dictionary with other dictionaries

        :Parameters:
            D : dict
                Main dictionary
            others : list/tuple/dict
                Other dictionary or composite dictionarized elements

        :Return:
            D : dict
                Merged dictionary
    """

    if type(others) == (list or tuple):
        for oth in others:
            D = {**D, **oth}

    elif type(others) == dict:
        D = {**D, **others}

    return D


def arraybin(arr, bins, method='mean'):
    """
    Resize an nD array by binning.

    :Parameters:
        arr : nD array
            N-dimensional array for binning.
        bins : list/tuple of int
            Bins/Size shrinkage along every axis.
        method : str
            Method for binning, 'mean' or 'sum'.

    :Return:
        arrbinned : nD array

    """

    bins = np.asarray(bins)
    nb = len(bins)

    if nb != arr.ndim:
        raise ValueError('Need to specify bins for all dimensions, use 1 for the dimensions not to be resized.')
    else:
        shape = np.asarray(arr.shape)
        binnedshape = shape // bins
        binnedshape = binnedshape.astype('int')

        # Calculate intermediate array shape
        shape_tuple = tuple(riffle(binnedshape, bins))
        # Calculate binning axis
        bin_axis_tuple = tuple(range(1, 2*nb+1, 2))

        if method == 'mean':
            arrbinned = arr.reshape(shape_tuple).mean(axis=bin_axis_tuple)
        elif method == 'sum':
            arrbinned = arr.reshape(shape_tuple).sum(axis=bin_axis_tuple)

        return arrbinned


def riffle(*arr):
    """
    Interleave multiple arrays of the same number of elements.

    :Parameter:
        *arr : array
            A number of arrays

    :Return:
        riffarr : 1D array
            An array with interleaving elements from each input array.
    """

    arr = (map(np.ravel, arr))
    arrlen = np.array(map(len, arr))

    try:
        unique_length = np.unique(arrlen).item()
        riffarr = np.vstack(arr).reshape((-1,), order='F')
        return riffarr
    except:
        raise ValueError('Input arrays need to have the same number of elements!')
