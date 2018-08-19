#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

# ========================= #
# Operations on point sets  #
# ========================= #

import numpy as np
from numpy.linalg import norm


def pointset_center(pset, condition='among', method='meancp'):
    """
    Determine the center position of a point set.

    :Parameters:
        pset : 2D array
            Pixel coordinates of the point set.
        condition : str | 'among'
            Condition to extract the points
            'among' = use a point among the set
            'unrestricted' = use the centroid coordinate
        method : str | 'meancp'
            Method to determine the point set center.
    """

    # Centroid position of point set
    pmean = np.mean(pset, axis=0)

    # Compare the coordinates with the mean position
    if method == 'meancp':
        dist = norm(pset - pmean, axis=1)
        minid = np.argmin(dist)
        pscenter = pset[minid, :]
        prest = np.delete(pset, minid, axis=0)
    else:
        raise NotImplementedError

    if condition == 'among':
        return pscenter, prest
    elif condition == 'unrestricted':
        return pmean


def order_pointset(pset, center=None, direction='cw'):
    """
    Order a point set around a center in a clockwise or counterclockwise way.

    :Parameters:
        pset : 2D array
            Pixel coordinates of the point set.
        center : list/tuple/1D array | None
            Pixel coordinates of the putative shape center.
        direction : str | 'cw'
            Direction of the ordering ('cw' or 'ccw').

    :Return:
        pset_ordered : 2D array
            Sorted pixel coordinates of the point set.
    """

    dirdict = {'cw':1, 'ccw':-1}

    # Calculate the coordinates of the
    if center is None:
        pmean = np.mean(pset, axis=0)
        pshifted = pset - pmean
    else:
        pshifted = pset - center

    pangle = np.arctan2(pshifted[:, 1], pshifted[:, 0]) * 180/np.pi
    # Sorting order
    order = np.argsort(pangle)[::dirdict[direction]]
    pset_ordered = pset[order]

    return pset_ordered
