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


def pointset_center(pset, condition='among', method='medcp'):
    """
    Determine the center position of a point set

    :Parameters:
        pset : 2D array
            Pixel coordinates of the point set.
        condition : str
            Condition to extract the points
            'among' = use a point among the set
            'unrestricted' = use the median coordinate
        method : str
            Method to determine the point set center.
    """

    med = np.median(pset[:, 0]), np.median(pset[:, 1])

    if method == 'medcp':
        dist = norm(pset - med, axis=1)
        pscenter = pset[np.argmin(dist), :]
    else:
        raise NotImplementedError

    if condition == 'among':
        return pscenter
    elif condition == 'unrestricted':
        return med


def order_polygon(pset, direction):

    return
