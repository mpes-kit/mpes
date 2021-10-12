#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

# ======================================= #
# Large facility beam time data analysis  #
# ======================================= #

from __future__ import print_function, division
import numpy as np
import numba as nb
from skimage.draw import circle, polygon


@nb.njit(parallel=False)
def _gridopt_landscape(u, v, shift_range, scale_range):
    """
    Calculate the optimization landscape.

    **Parameters**\n
    u, v: numpy.ndarray
        Vectors to compare in the grid search.
    shift_range: numpy.ndarray (Lshift x 1)
        Array containing the shift values to evaluate.
    scale_range: numpy.ndarray (Lscale x 1)
        Array containing the scale values to evaluate.

    **Return**\n
    vopt: numpy.ndarray (Lshift x Lscale)
        Numerical optimization landscape obtained from grid search.
    """

    lshift = shift_range.size
    lscale = scale_range.size

    vopt = np.zeros((lshift, lscale))

    for si, s in enumerate(shift_range):
        for ai, a in enumerate(scale_range):

            # Figure of merit for the overlap between two 1D signals
            vopt[si, ai] = np.nansum((u[:-s] - a*v[s:])**2)

    return vopt


def planarfilter(U, axis, leftshift=0, rightshift=1, upshift=0, downshift=1, shifts=None):
    """
    An nearest neighbor filter on 3D data.

    **Parameters**\n
    U: numpy.ndarray
        3D matrix for spatial filtering.
    axis: int
        Axis perpendicular to the spatial domain.
    leftshift, rightshift, upshift, downshift: int | 0, 1, 0, 1
        Shift parameters along the four principal directions.
        left-right range : pixel - leftshift -- pixel + rightshift
        top-down range : pixel - upshift -- pixel + downshift
    shifts: list of numerics
        Collection of shift parameters along the four directions
        (overwrites the separate assignment).

    **Return**\n
    V: numpy.ndarray
        Matrix output after nearest-neighbor spatial averaging.
    """

    # Gather shift parameters
    if shifts is not None:
        lsh, rsh, ush, dsh = shifts

    V = U.copy()
    V = np.moveaxis(V, axis, 0)
    V = nnmean(U, V, lsh, rsh, ush, dsh)
    V = np.moveaxis(V, 0, axis)

    return V


@nb.njit('float32[:,:,:](float32[:,:,:], float32[:,:,:], int64, int64, int64, int64)', parallel=True)
def nnmean(U, V, lsh, rsh, ush, dsh):
    """
    Nearest-neighbor mean averaging

    **Parameters**\n
    U, V: numpy.ndarray (float32)
        3D matrices, U is the original, V is the modified version of U.
    lsh, rsh, ush, dsh: int
        Pixel shifts along the four primary directions (left, right, up, down).

    **Return**\n
    V: numpy.ndarray (float32)
        Modified 3D matrix after averaging.
    """

    a, x, y = V.shape

    for i in nb.prange(ush, x-dsh):
        for j in nb.prange(lsh, y-rsh):
            for ia in nb.prange(a):

                # Average over nearby points perpendicular to the specified axis
                V[ia, i, j] = np.nanmean(U[ia, i-ush:i+dsh, j-lsh:j+rsh])

    return V


def calcShiftScale(U, V, axis, **kwds):
    """
    Calculate the shift and scale matrices that aligns the matrix V to U by grid search.

    **Parameters**\n
    U, V: numpy.ndarray
        3D matrices for alignment (from V to U).
    axis: int
        The axis along which to align V to U matrices.
    rgshift: numpy.ndarray
        The range of shifts to iterate over.
    rgscale: numpy.ndarray
        The range of scales to iterate over.

    **Returns**\n
    shift: numpy.ndarray
        The matrix of optimal shifts to align V to U.
    scale: numpy.ndarray
        The matrix of optimal scales to align V to U.
    """

    rgshift = kwds.pop('shifts', np.linspace(1, 10, 10, dtype='int32'))
    rgscale = kwds.pop('scales', np.linspace(0.5, 1.5, 40, dtype='float32'))

    # Permute the alignment axis to the first position of the dimensions
    U = np.moveaxis(U, axis, 0)
    V = np.moveaxis(V, axis, 0)

    zeromat = np.zeros(U.shape[1:], dtype='float32')

    shift, scale = _shiftscale(U, V, rgshift, rgscale, zeromat)

    return shift, scale


@nb.njit('Tuple((float32[:,:], float32[:,:]))(float32[:,:,:], float32[:,:,:], int32[:], float32[:], float32[:,:])', parallel=True)
def _shiftscale(U, V, rgshift, rgscale, zeromat):
    """
    Calculation of the shift and scale matrix (jit-version)

    In case of multiple identical minima in the grid search landscape,
    the first is selected.
    """

    x, y = U.shape[1:]
    #shift, scale = np.zeros((x, y), dtype='float32'), np.zeros((x, y), dtype='float32')
    shift, scale = zeromat.copy(), zeromat.copy()

    for i in nb.prange(x):
        for j in nb.prange(y):

            u, v = U[:, i, j], V[:, i, j]
            grid = _gridopt_landscape(u, v, rgshift, rgscale)

            minr, minc = np.where(grid == np.min(grid))
            shift[i, j], scale[i, j] = rgshift[minr[0]], rgscale[minc[0]]

    return shift, scale


def applyAlignment(V, shift, scale, axis=2, filterkwd=None, ret='mat'):
    """
    Apply the calculated shift and scale matrices to the volume using the
    formula, W = scale * shift * V

    **Parameters**\n
    V: numpy.ndarray
        The 3D matrix to adjust.
    shift: numpy.ndarray
        The shift matrix.
    scale: numpy.ndarray
        The scale matrix.
    axis: int
        The axis to apply the shift matrix to.
    filterkwd: dict
        Keyword arguments to supply to ``mpes.beamtime.planarfilter()``.

    **Return**\n
    trimmed W matrix: numpy.ndarray
        3D matrix after application of alignment (different size from V)
    """

    shift = shift.astype('int')

    # Apply spatial filtering to the band structure data, if specified
    if filterkwd:
        V = planarfilter(V, **filterkwd)

    V = np.moveaxis(V, axis, 0)
    nr, nc = V.shape[1:]

    # Apply the scale matrix
    V *= scale[None,...]
    W = V.copy()

    # Apply the shift matrix
    maxshift = np.max(shift)
    for r in range(nr):
        for c in range(nc):
            s = shift[r, c]
            W[:-s, r, c] = V[s:, r, c]

    W = W[:-maxshift,...]
    W = np.moveaxis(W, 0, axis)

    if ret == 'all':
        return W, maxshift
    elif ret == 'mat':
        return W
