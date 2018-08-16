#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

import numpy as np
import cv2
from xarray import DataArray
from . import fprocessing as fp


class BandStructure(DataArray):
    """
    Data structure for storage and manipulation of a single band structure (1-3D) dataset.
    """

    def __init__(self, faddr=None, data=None, coords=None, dims=None, datakey='V', **kwds):

        self.faddr = faddr

        # Initialization by loading data from an hdf5 file
        if self.faddr is not None:
            hdfdict = fp.readBinnedhdf5(self.faddr)
            data = hdfdict.pop(datakey)
            super().__init__(data, coords=hdfdict, dims=hdfdict.keys(), **kwds)
        # Initialization by direct connection to existing data
        elif self.faddr is None:
            super().__init__(data, coords=coords, dims=dims, **kwds)

    @staticmethod
    def rotate(data, axis):
        pass

    @staticmethod
    def intensityTransform(data, axis, scale_matrix=None):
        """
        Scaling and masking of band structure data
        """
        pass


class MPESDataset(BandStructure):
    """
    Data structure for storage and manipulation of a multidimensional photoemission
    spectroscopy (MPES) dataset (4D and above).
    """

    def __init__(self, faddr=None, data=None, coords=None, dims=None, datakey='V', **kwds):

        self.faddr = faddr

        # Initialization by loading data from an hdf5 file
        if self.faddr is not None:
            hdfdict = fp.readBinnedhdf5(self.faddr)
            data = hdfdict.pop(datakey)
            super().__init__(data, coords=hdfdict, dims=hdfdict.keys(), **kwds)
        # Initialization by direct connection to existing data
        elif self.faddr is None:
            super().__init__(data, coords=coords, dims=dims, **kwds)

    def subset(self, axis, axisrange):
        """
        Spawn instances of BandStructure class from axis slicing
        """

        if axis == 'tpp':

            bs = BandStructure()

            return bs
