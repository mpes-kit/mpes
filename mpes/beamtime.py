#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

# =======================================
# Beam time data analysis
# =======================================

from __future__ import print_function, division
import numpy as np
import numba as nb
from skimage.draw import circle, polygon


def planarfilter(U, axis, leftshift=0, rightshift=1, upshift=0, downshift=1, shifts=None):
    """
    An nearest neighbor filter on 3D data

    :Parameters:
        U : numpy.ndarray
            3D matrix for spatial filtering.
        axis : int
            Axis perpendicular to the spatial domain.
        leftshift, rightshift, upshift, downshift : int | 0, 1, 0, 1
            Shift parameters along the four principal directions.
            left-right range : pixel - leftshift -- pixel + rightshift
            top-down range : pixel - upshift -- pixel + downshift
        shifts : list of numerics
            Collection of shift parameters along the four directions
            (overwrites the separate assignment).

    :Return:
        V : numpy.ndarray
            Matrix output after nearest-neighbor spatial averaging
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

    :Parameters:
        U, V : numpy.ndarray (float32)
            3D matrices, U is the original, V is the modified version of U.
        lsh, rsh, ush, dsh : int
            Pixel shifts along the four primary directions (left, right, up, down).

    :Return:
        V : numpy.ndarray (float32)
            Modified 3D matrix after averaging.
    """

    a, x, y = V.shape

    for i in nb.prange(ush, x-dsh):
        for j in nb.prange(lsh, y-rsh):
            for ia in nb.prange(a):

                # Average over nearby points perpendicular to the specified axis
                V[ia, i, j] = np.nanmean(U[ia, i-ush:i+dsh, j-lsh:j+rsh])

    return V
