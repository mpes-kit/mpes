#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

##########################################
# Dask-compatible thin-plate spline (TPS)
##########################################

#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import dask.array as da

_small = 1e-10


def _U(x):
    """
    Compute the U function as part of the thin-plate function.
    """

    return (x**2) * np.where(x<_small, 0, np.log(x))


def _U_dask(col):
    """
    Compute the U function as part of the thin-plate function.

    **Parameter**\n
    col: dask.Series or numpy.ndarray
        A column vector of ``dask`` (or ``pandas``) ``Series``.
    """

    return col**2 * da.log(col.abs())


def _calculate_f(coeffs, points, x, y, sumcol, type='dask'):
    """
    Calculate the thin-plate energy function.
    """

    w = coeffs[:-3]
    a1, ax, ay = coeffs[-3:]

    for wi, Pi in zip(w, points):
        sumcol += wi * _U_dask(da.sqrt((x-Pi[0])**2 + (y-Pi[1])**2))

    return a1 + ax*x + ay*y + sumcol


def _interpoint_distances(points):
    """
    Calculate the pair distance within a point set.
    """

    xd = np.subtract.outer(points[:,0], points[:,0])
    yd = np.subtract.outer(points[:,1], points[:,1])

    return np.sqrt(xd**2 + yd**2)


def _make_L_matrix(points):
    """
    Construct the L matrix following Bookstein's description.
    """

    n = len(points)
    K = _U(_interpoint_distances(points))
    P = np.ones((n, 3))
    P[:,1:] = points
    O = np.zeros((3, 3))
    # Construct L matrix from constituent blocks
    L = np.asarray(np.bmat([[K, P], [P.transpose(), O]]))

    return L


def tps_coeffs(from_points, to_points):
    """
    Compute thin-plate spline coefficients.
    """

    from_points, to_points = np.asarray(from_points), np.asarray(to_points)
    err = np.seterr(divide='ignore')
    L = _make_L_matrix(from_points)

    V = np.resize(to_points, (len(to_points)+3, 2))
    V[-3:, :] = 0
    coeffs = np.dot(np.linalg.pinv(L), V)
    np.seterr(**err)

    return coeffs
