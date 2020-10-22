#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from math import cos, pi
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm_notebook
from tqdm import tqdm as tqdm_classic


def find_nearest(val, narray):
    """
    Find the value closest to a given one in a 1D array.

    **Parameters**\n
    val: float
        Value of interest.
    narray: 1D numeric array
        The array to look for the nearest value.

    **Return**\n
    ind: int
        Array index of the value nearest to the given one.
    """

    return np.argmin(np.abs(narray - val))


def numFormatConversion(seq, form='int', **kwds):
    """
    When length keyword is not specified as an argument, the function
    returns a format-converted sequence of numbers. The function returns
    nothing when the conversion fails due to errors.

    **Parameters**\n
    seq: 1D numeric array
        The numeric array to be converted.
    form: str | 'int'
        The format to convert into.

    **Return**\n
    numseq: converted numeric type
        The format-converted array.
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
    Convert a single number to its nearest odd number.

    **Parameters**\n
    num: float/int
        Number to convert.

    **Return**\n
    oddnum: int
        The nearest odd number.
    """

    rnum = int(num)
    rem = rnum % 2
    oddnum = rnum + int(cos(rem*pi/2))

    return oddnum


def intify(*nums):
    """ Safely convert to integer (avoiding None).

    **Parameter**\n
    nums: list/tuple/1D array
        Numeric array to convert to integer.

    **Return**\n
    intnums : list
        Converted list of numerics.
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
    Reverse an ndarray along certain axis.

    **Parameters**\n
    arr: nD numeric array
        array to invert
    axis: int | -1
        the axis along which to invert

    **Return**\n
    revarr: nD numeric array
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

    **Parameters**\n
    entry: numeric/str
        Repeated item in nested list.
    row: int
        Number of rows in nested list.
    column: int
        Number of columns in nested list.

    **Return**\n
        Nested list.
    """

    return [[entry]*column for _ in range(row)]


def normspec(*specs, smooth=False, span=13, order=1):
    """
    Normalize a series of 1D signals.

    **Parameters**\n
    *specs: list/2D array
        Collection of 1D signals.
    smooth: bool | False
        Option to smooth the signals before normalization.
    span, order: int, int | 13, 1
        Smoothing parameters of the LOESS method (see ``scipy.signal.savgol_filter()``).

    **Return**\n
    normalized_specs: 2D array
        The matrix assembled from a list of maximum-normalized signals.
    """

    nspec = len(specs)
    specnorm = []

    for i in range(nspec):

        spec = specs[i]

        if smooth:
            spec = savgol_filter(spec, span, order)

        if type(spec) in (list, tuple):
            nsp = spec / max(spec)
        else:
            nsp = spec / spec.max()
        specnorm.append(nsp)

        # Align 1D spectrum
        normalized_specs = np.asarray(specnorm)

    return normalized_specs


def appendformat(filepath, form):
    """
    Append a format string to the end of a file path.

    **Parameters**\n
    filepath: str
        File path of interest.
    form: str
        File format of interest.
    """

    format_string = '.'+form
    if filepath:
        if not filepath.endswith(format_string):
            filepath += format_string

    return filepath


def shuffleaxis(arr, axes, direction='front'):
    """
    Move multiple axes of a multidimensional array simultaneously
    to the front or end of its axis order.

    **Parameters**\n
    arr: ndarray
        Array to be shuffled.
    axes: tuple of int
        Dimensions to be shuffled.
    direction: str | 'front'
        Direction of shuffling ('front' or 'end').

    **Return**\n
    sharr: ndarray
        Dimension-shuffled array.
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
    """
    Merge a dictionary with other dictionaries.

    **Parameters**\n
    D: dict
        Main dictionary.
    others: list/tuple/dict
        Other dictionary or composite dictionarized elements.

    **Return**\n
    D: dict
        Merged dictionary.
    """

    if type(others) in (list, tuple): # Merge D with a list or tuple of dictionaries
        for oth in others:
            D = {**D, **oth}

    elif type(others) == dict: # Merge D with a single dictionary
        D = {**D, **others}

    return D


def riffle(*arr):
    """
    Interleave multiple arrays of the same number of elements.

    **Parameter**\n
    *arr: array
        A number of arrays.

    **Return**\n
    riffarr: 1D array
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


def arraybin(arr, bins, method='mean'):
    """
    Resize an nD array by binning.

    **Parameters**\n
    arr: nD array
        N-dimensional array for binning.
    bins: list/tuple of int
        Bins/Size shrinkage along every axis.
    method: str
        Method for binning, 'mean' or 'sum'.

    **Return**\n
    arrbinned: numpy array
        Array after resizing.

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


def calcax(start, end, steps, ret='midpoint'):
    """ Calculate the positions of the axes values.

    **Parameters**\n
    start, end, steps: numeric, numeric, numeric
        Start, end positions and the steps.
    ret: str | 'midpoints'
        Specification of positions.
    """

    edges = np.linspace(start, end, steps+1, endpoint=True)

    if ret == 'edge':
        return edges

    elif ret == 'midpoint':
        midpoints = (edges[1:] + edges[:-1])/2
        return midpoints


def bnorm(pval, pmax, pmin):
    """ Normalize parameters by the bounds of the given values.

    **Parameters**\n
    pval: array/numeric
        A single value/collection of values to normalize.
    pmax, pmin: numeric, numeric
        The maximum and the minimum of the values.

    **Return**\n
        Normalized values (with the same dimensions as pval).

    """

    return (pval - pmin) / (pmax - pmin)


def tqdmenv(env):
    """ Choose tqdm progress bar executing environment.

    **Parameter**\n
    env: str
        Name of the environment, 'classic' for ordinary environment,
        'notebook' for Jupyter notebook.
    """

    if env == 'classic':
        tqdm = tqdm_classic
    elif env == 'notebook':
        tqdm = tqdm_notebook

    return tqdm


def concat(*arrays):
    """ Concatenate a sequence of (up to 2D) array-like objects along a given axis.
    """

    array_list = []
    for ia, array in enumerate(arrays):
        arrdim = np.ndim(array)

        if arrdim == 1:
            array = np.asarray(array)[None,...]

        array_list.append(array)

    return np.concatenate(array_list, axis=0)


def multithresh(arr, lbs, ubs, ths):
    """ Multilevel thresholding of a 1D array. Somewhat similar to bit depth reduction.

    **Parameters**\n
    arr: 1D array
        Array for thresholding.
    lbs, ubs: list/tuple/array, list/tuple/array
        Paired lower and upper bounds for each thresholding level.
    ths: list/tuple/array
        Thresholds for the values within the paired lower and upper bounds.
    """

    for lb, ub, th in zip(lbs, ubs, ths):
        if (arr > lb) & (arr < ub):
            return th
