#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

import numpy as np
import cv2
from copy import deepcopy
from xarray import DataArray
from mpes import fprocessing as fp, analysis as aly, utils as u


class BandStructure(DataArray):
    """
    Data structure for storage and manipulation of a single band structure (1-3D) dataset.
    Instantiation of the BandStructure class can be done by specifying a (HDF5 or mat) file path
    or by separately specify the data, the axes values and their names.
    """

    def __init__(self, data=None, coords=None, dims=None, datakey='V', faddr=None, typ='float32', **kwds):

        self.faddr = faddr
        # Specify the symmetries of the band structure
        self.rot_sym_order = kwds.pop('rot_sym_order', 1) # Lowest rotational symmetry
        self.mir_sym_order = kwds.pop('mir_sym_order', 0) # No mirror symmetry

        # Initialization by loading data from an hdf5 file (details see mpes.fprocessing)
        if self.faddr is not None:
            hdfdict = fp.readBinnedhdf5(self.faddr, typ=typ)
            data = hdfdict.pop(datakey)
            self.axesdict = hdfdict
            super().__init__(data, coords=hdfdict, dims=hdfdict.keys(), **kwds)

        # Initialization by direct connection to existing data
        elif self.faddr is None:
            self.axesdict = coords
            super().__init__(data, coords=coords, dims=dims, **kwds)

        #setattr(self.data, 'datadim', self.data.ndim)
        #self['datadim'] = self.data.ndim

    def kcenter_estimate(self, threshold, dimname='E', view=False):
        """
        Estimate the momentum center of the isoenergetic plane.
        """

        if dimname not in self.coords.keys():
            raise ValueError('Need to specify the name of the energy dimension if different from default (E)!')
        else:
            center = (0, 0)

            if view:
                pass

            return center

    def scale(self, axis, scale_array, update=True, ret=False):
        """
        Scaling and masking of band structure data.

        :Parameters:
            axis : str/tuple
                Axes along which to apply the intensity transform.
            scale_array : nD array
                Scale array to be applied to data.
            update : bool | True
                Options to update the existing array with the intensity-transformed version.
            ret : bool | False
                Options to return the intensity-transformed data.

        :Return:
            itdata : nD array
                Intensity-transformed data.
        """

        itdata = aly.apply_mask_along(self.data, mask=scale_array, axes=axis)

        if update:
            self.data = itdata

        if ret:
            return itdata

    def resize(self, factor, method='mean', update=True, ret=False):
        """
        Reduce the size of the axis through rebinning.

        :Parameters:
            factor : list/tuple of int
                Resizing factor for each dimension (e.g. 2 means reduce by a factor of 2).
            method : str | 'mean'
                Numerical operation used for resizing ('mean' or 'sum').
            update : bool | False
                Option to update current dataset.
            ret : bool | False
                Option to return the resized data array.

        :Return:
            binarr : nD array
                Resized n-dimensional array.
        """

        binarr = u.arraybin(self.data, factor, method=method)

        if update:
            self.data = binarr
            # Update axis values

        if ret:
            return binarr

    def rotate(data, axis, update=True, ret=False):
        """
        Primary axis rotation.
        """

        # Slice out
        rdata = np.moveaxis(self.data, axis, 0)
        #data =
        rdata = np.moveaxis(self.data, 0, axis)

        if update:
            self.data = rdata
            # No change of axis values

        if ret:
            return rdata

    def orthogonalize(self, center, update=True, ret=False):
        """
        Align the high symmetry axes in the isoenergetic plane to the row and
        column directions of the image coordinate system.
        """

        pass

    def symmetrize(self, center, symtype, update=True, ret=False):
        """
        Symmetrize data within isoenergetic planes. Supports rotational and
        mirror symmetries.
        """

        if symtype == 'mirror':
            pass
        elif symtype == 'rotational':
            pass

        if update:
            pass

        if ret:
            return

    def _view_result(self):
        """
        2D visualization of temporary result.
        """

        pass


class MPESDataset(BandStructure):
    """
    Data structure for storage and manipulation of a multidimensional photoemission
    spectroscopy (MPES) dataset (4D and above).
    """

    def __init__(self, data=None, coords=None, dims=None, datakey='V', faddr=None, typ='float32', **kwds):

        self.faddr = faddr

        # Initialization by loading data from an hdf5 file
        if self.faddr is not None:
            hdfdict = fp.readBinnedhdf5(self.faddr, combined=True, typ=typ)
            data = hdfdict.pop(datakey)
            self.axesdict = hdfdict
            super().__init__(data, coords=hdfdict, dims=hdfdict.keys(), **kwds)

        # Initialization by direct connection to existing data
        elif self.faddr is None:
            self.axesdict = coords
            super().__init__(data, coords=coords, dims=dims, **kwds)

    def gradient(self):

        pass

    def maxdiff(self):
        """
        Find the hyperslice with maximum difference from the specified one.
        """

        pass

    def subset(self, axis, axisrange):
        """
        Spawn instances of BandStructure class from axis slicing
        """

        if axis == 'tpp':

            axid = self.get_axis_num('tpp')
            subdata = np.moveaxis(self.data, axid, 0)

            try:
                subdata = subdata[axisrange,...].mean(axis=0)
            except:
                subdata = subdata[axisrange,...]

            # Copy the correct axes values after slicing
            tempdict = deepcopy(self.axesdict)
            tempdict.pop(axis)

            bs = BandStructure(subdata, coords=tempdict, dims=tempdict.keys())

            return bs
