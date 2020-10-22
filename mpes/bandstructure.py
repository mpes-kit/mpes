#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

# ======================================================= #
# Main data structure for processing band structure data  #
# ======================================================= #

from __future__ import print_function, division
from . import fprocessing as fp, analysis as aly, utils as u, visualization as vis
import numpy as np
import cv2
from copy import deepcopy
from xarray import DataArray
from symmetrize import pointops as po
from collections import OrderedDict


class BandStructure(DataArray):
    """
    Data structure for storage and manipulation of a single band structure (1-3D) dataset.
    Instantiation of the BandStructure class can be done by specifying a (HDF5 or mat) file path
    or by separately specify the data, the axes values and their names.
    """

    keypair = OrderedDict({'ADC':'tpp', 'X':'kx', 'Y':'ky', 't':'E'})

    def __init__(self, data=None, coords=None, dims=None, datakey='V', faddr=None, typ='float32', **kwds):

        self.faddr = faddr
        self.axesdict = OrderedDict() # Container for axis coordinates

        # Specify the symmetries of the band structure
        self.rot_sym_order = kwds.pop('rot_sym_order', 1) # Lowest rotational symmetry
        self.mir_sym_order = kwds.pop('mir_sym_order', 0) # No mirror symmetry
        self.kcenter = [0, 0]
        self.high_sym_points = []
        self.sym_points_dict = {}

        # Initialization by loading data from an hdf5 file (details see mpes.fprocessing)
        if self.faddr is not None:

            hdfdict = fp.readBinnedhdf5(self.faddr, typ=typ)
            data = hdfdict.pop(datakey)

            for k, v in self.keypair.items():
                # When the file already contains the converted axes, read in directly
                try:
                    self.axesdict[v] = hdfdict[v]
                # When the file contains no converted axes, rename coordinates according to the keypair correspondence
                except:
                    self.axesdict[v] = hdfdict[k]

            super().__init__(data, coords=self.axesdict, dims=self.axesdict.keys(), **kwds)

        # Initialization by direct connection to existing data
        elif self.faddr is None:

            self.axesdict = coords
            super().__init__(data, coords=coords, dims=dims, **kwds)

        #setattr(self.data, 'datadim', self.data.ndim)
        #self['datadim'] = self.data.ndim

    def keypoint_estimate(self, img, dimname='E', pdmethod='daofind', display=False, update=False, ret=False, **kwds):
        """
        Estimate the positions of momentum local maxima (high symmetry points) in the isoenergetic plane.
        """

        if dimname not in self.coords.keys():
            raise ValueError('Need to specify the name of the energy dimension if different from default (E)!')

        else:
            direction = kwds.pop('direction', 'cw')
            pks = po.peakdetect2d(img, method=pdmethod, **kwds)

            # Select center and non-center peaks
            center, verts = po.pointset_center(pks)
            hsp = po.order_pointset(verts, direction=direction)

            if update:
                self.center = center
                self.high_sym_points = hsp

            if display:
                self._view_result(img)
                for ip, p in enumerate(hsp):
                    self['ax'].scatter(p[1], p[0], s=20, c='k')
                    self['ax'].text(p[1]+3, p[0]+3, str(ip), c='r')
                self['ax'].text(center[0], center[1], 'C', color='r')

            if ret:
                return center, hsp

    def scale(self, axis, scale_array, update=True, ret=False):
        """
        Scaling and masking of band structure data.

        **Parameters**\n
        axis: str/tuple
            Axes along which to apply the intensity transform.
        scale_array: nD array
            Scale array to be applied to data.
        update: bool | True
            Options to update the existing array with the intensity-transformed version.
        ret: bool | False
            Options to return the intensity-transformed data.

        **Return**\n
        scdata: nD array
            Data after intensity scaling.
        """

        scdata = aly.apply_mask_along(self.data, mask=scale_array, axes=axis)

        if update:
            self.data = scdata

        if ret:
            return scdata

    def update_axis(self, axes=None, vals=None, axesdict=None):
        """
        Update the values of multiple axes.

        **Parameters**\n
        axes: list/tuple | None
            Collection of axis names.
        vals: list/tuple | None
            Collection of axis values.
        axesdict: dict | None
            Axis-value pair for update.
        """

        if axesdict:
            self.coords.update(axesdict)
        else:
            axesdict = dict(zip(axes, vals))
            self.coords.update(axesdict)

    @classmethod
    def resize(cls, data, axes, factor, method='mean', ret=True, **kwds):
        """
        Reduce the size (shape-changing operation) of the axis through rebinning.

        **Parameters**\n
        data: nD array
            Data to resize (e.g. self.data).
        axes: dict
            Axis values of the original data structure (e.g. self.coords).
        factor: list/tuple of int
            Resizing factor for each dimension (e.g. 2 means reduce by a factor of 2).
        method: str | 'mean'
            Numerical operation used for resizing ('mean' or 'sum').
        ret: bool | False
            Option to return the resized data array.

        **Return**\n
            Instance of resized n-dimensional array along with downsampled axis coordinates.
        """

        binarr = u.arraybin(data, factor, method=method)

        axesdict = OrderedDict()
        # DataArray sizes cannot be changed, need to create new class instance
        for i, (k, v) in enumerate(axes.items()):
            fac = factor[i]
            axesdict[k] = v[::fac]

        if ret:
            return cls(data=binarr, coords=axesdict, dims=axesdict.keys(), **kwds)

    def rotate(data, axis, angle, angle_unit='deg', update=True, ret=False):
        """
        Primary axis rotation that preserves the data size.
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
        mirror symmetries. The operation preserves the data size.
        """

        if symtype == 'mirror':
            pass
        elif symtype == 'rotational':
            pass

        if update:
            pass

        if ret:
            return

    def _view_result(self, img, figsize=(5,5), cmap='terrain_r', origin='lower'):
        """
        2D visualization of intermediate result.
        """

        self['fig'], self['ax'] = plt.subplots(figsize=figsize)
        self['ax'].imshow(img, cmap=cmap, origin=origin)

    def saveas(self, form='h5', save_addr='./'):

        pass


class MPESDataset(BandStructure):
    """
    Data structure for storage and manipulation of a multidimensional photoemission
    spectroscopy (MPES) dataset (4D and above).
    """

    def __init__(self, data=None, coords=None, dims=None, datakey='V', faddr=None, typ='float32', **kwds):

        self.faddr = faddr
        self.f, self.ax = None, None
        self.axesdict = OrderedDict()

        # Initialization by loading data from an hdf5 file
        if self.faddr is not None:
            hdfdict = fp.readBinnedhdf5(self.faddr, combined=True, typ=typ)
            data = hdfdict.pop(datakey)

            # Add other key pairs to the instance
            otherkp = kwds.pop('other_keypair', None)
            if otherkp:
                self.keypair = u.dictmerge(self.keypair, otherkp)

            for k, v in self.keypair.items():
                # When the file already contains the converted axes, read in directly
                try:
                    self.axesdict[v] = hdfdict[v]
                # When the file contains no converted axes, rename coordinates according to the keypair correspondence
                except:
                    self.axesdict[v] = hdfdict[k]

            super().__init__(data, coords=self.axesdict, dims=self.axesdict.keys(), **kwds)

        # Initialization by direct connection to existing data
        elif self.faddr is None:
            self.axesdict = coords
            super().__init__(data, coords=coords, dims=dims, **kwds)

    def slicediff(self, slicea, sliceb, slicetype='index', axreduce=None, ret=False, **kwds):
        """
        Calculate the difference of two hyperslices (hs), hsa - hsb.

        **Parameters**\n
        slicea, sliceb: dict
            Dictionaries for slicing.
        slicetype: str | 'index'
            Type of slicing, 'index' (DataArray.isel) or 'value' (DataArray.sel).
        axreduce: tuple of int | None
            Axes to sum over.
        ret: bool | False
            Options for return.
        **kwds: keyword arguments
            Those passed into DataArray.isel() and DataArray.sel().

        **Return**\n
        sldiff: class
            Sliced class instance.
        """

        drop = kwds.pop('drop', False)

        # Calculate hyperslices
        if slicetype == 'index': # Index-based slicing

            sla = self.isel(**slicea, drop=drop)
            slb = self.isel(**sliceb, drop=drop)

        elif slicetype == 'value': # Value-based slicing

            meth = kwds.pop('method', None)
            tol = kwds.pop('tol', None)

            sla = self.sel(**slicea, method=meth, tolerance=tol, drop=drop)
            slb = self.sel(**sliceb, method=meth, tolerance=tol, drop=drop)

        # Calculate the difference between hyperslices
        if axreduce:
            sldiff = sla.sum(axis=axreduce) - slb.sum(axis=axreduce)

        else:
            sldiff = sla - slb

        if ret:
            return sldiff

    def maxdiff(self, vslice, ret=False):
        """
        Find the hyperslice with maximum difference from the specified one.
        """

        raise NotImplementedError

    def subset(self, axis, axisrange):
        """
        Spawn an instance of the BandStructure class from axis slicing.

        **Parameters**\n
        axis: str/list
            Axes to subset from.
        axisrange: slice object/list
            The value range of axes to be sliced out.

        **Return**\n
            An instances of ``BandStructure``, ``MPESDataset`` or ``DataArray`` class.
        """

        # Determine the remaining coordinate keys using set operations
        restaxes = set(self.coords.keys()) - set(axis)
        bsaxes = set(['kx', 'ky', 'E'])

        # Construct the subset data and the axes values
        axid = self.get_axis_num(axis)
        subdata = np.moveaxis(self.data, axid, 0)

        try:
            subdata = subdata[axisrange,...].mean(axis=0)
        except:
            subdata = subdata[axisrange,...]

        # Copy the correct axes values after slicing
        tempdict = deepcopy(self.axesdict)
        tempdict.pop(axis)

        # When the remaining axes are only a set of (kx, ky, E),
        # Create a BandStructure instance to contain it.
        if restaxes == bsaxes:
            bs = BandStructure(subdata, coords=tempdict, dims=tempdict.keys())

            return bs

        # When the remaining axes contain a set of (kx, ky, E) and other parameters,
        # Return an MPESDataset instance to contain it.
        elif bsaxes < restaxes:
            mpsd = MPESDataset(subdata, coords=tempdict, dims=tempdict.keys())

            return mpsd

        # When the remaining axes don't contain a full set of (kx, ky, E),
        # Create a normal DataArray instance to contain it.
        else:
            dray = DataArray(subdata, coords=tempdict, dims=tempdict.keys())

            return dray

    def saveas(self, form='h5', save_addr='./'):

        pass
