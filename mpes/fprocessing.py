#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""
# =========================
# Sections:
# 1.  Utility functions
# 2.  File I/O and parsing
# 3.  Data transformation
# =========================

from __future__ import print_function, division
from .igoribw import loadibw
from .base import FileCollection, MapParser, saveClassAttributes
from . import utils as u, bandstructure as bs
import igor.igorpy as igor
import pandas as pd
import os
import re
import glob as g
import numpy as np
import numpy.fft as nft
import scipy.io as sio
import skimage.io as skio
from PIL import Image as pim
import warnings as wn
from h5py import File
import psutil as ps
import dask as d, dask.array as da, dask.dataframe as ddf
from dask.diagnostics import ProgressBar
import natsort as nts
from functools import reduce
from funcy import project

N_CPU = ps.cpu_count()

# ================= #
# Utility functions #
# ================= #

def sgfltr2d(datamat, span, order, axis=0):
    """
    Savitzky-Golay filter for two dimensional data
    Operated in a line-by-line fashion along one axis
    Return filtered data
    """

    dmat = np.rollaxis(datamat, axis)
    r, c = np.shape(datamat)
    dmatfltr = np.copy(datamat)
    for rnum in range(r):
        dmatfltr[rnum, :] = savgol_filter(datamat[rnum, :], span, order)

    return np.rollaxis(dmatfltr, axis)


def sortNamesBy(namelist, pattern, gp=0, slicerange=(None, None)):
    """
    Sort a list of names according to a particular sequence of numbers
    (specified by a regular expression search pattern)

    Parameters

    namelist : str
        List of name strings
    pattern : str
        Regular expression of the pattern
    gp : int
        Grouping number

    Returns

    orderedseq : array
        Ordered sequence from sorting
    sortednamelist : str
        Sorted list of name strings
    """

    gp = int(gp)
    sa, sb = slicerange

    # Extract a sequence of numbers from the names in the list
    seqnum = np.array([re.search(pattern, namelist[i][sa:sb]).group(gp)
                       for i in range(len(namelist))])
    seqnum = seqnum.astype(np.float)

    # Sorted index
    idx_sorted = np.argsort(seqnum)

    # Sort the name list according to the specific number of interest
    sortednamelist = [namelist[i] for i in idx_sorted]

    # Return the sorted number sequence and name list
    return seqnum[idx_sorted], sortednamelist


def rot2d(th, angle_unit):
    """
    construct 2D rotation matrix
    """

    if angle_unit == 'deg':
        thr = np.deg2rad(th)
        return np.array([[np.cos(thr), -np.sin(thr)],
                         [np.sin(thr), np.cos(thr)]])

    elif angle_unit == 'rad':
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


# ====================== #
#  File I/O and parsing  #
# ====================== #

def readimg(f_addr):
    """
    Read images (jpg, png, 2D/3D tiff)
    """

    return skio.imread(f_addr)


def readtsv(fdir, header=None, dtype='float', **kwds):
    """
    Read tsv file from hemispherical detector

    **Parameters**

    fdir : str
        file directory
    header : int | None
        number of header lines
    dtype : str | 'float'
        data type of the return numpy.ndarray
    **kwds : keyword arguments
        other keyword arguments for pandas.read_table()

    **Return**

    data : numpy ndarray
        read and type-converted data
    """

    data = np.asarray(pd.read_table(fdir, delim_whitespace=True,
                        header=None, **kwds), dtype=dtype)
    return data


def readIgorBinFile(fdir, **kwds):
    """
    Read Igor binary formats (pxp and ibw)
    """

    ftype = kwds.pop('ftype', fdir[-3:])
    errmsg = "Error in file loading, please check the file format."

    if ftype == 'pxp':

        try:
            igfile = igor.load(fdir)
        except IOError:
            print(errmsg)

    elif ftype == 'ibw':

        try:
            igfile = loadibw(fdir)
        except IOError:
            print(errmsg)

    else:

        raise IOError(errmsg)

    return igfile


def readARPEStxt(fdir, withCoords=True):
    """
    Read and convert Igor-generated ARPES .txt files into numpy arrays
    The withCoords option specify whether the energy and angle information is given
    """

    if withCoords:

        # Retrieve the number of columns in the txt file
        dataidx = pd.read_table(fdir, skiprows=1, header=None).columns
        # Read all data with the specified columns
        datamat = pd.read_table(fdir, skiprows=0, header=None, names=dataidx)
        # Shift the first row by one value (align the angle axis)
        #datamat.iloc[0] = datamat.iloc[0].shift(1)

        ARPESData = datamat.loc[1::, 1::].values
        EnergyData = datamat.loc[1::, 0].values
        AngleData = datamat.loc[0, 1::].values

        return ARPESData, EnergyData, AngleData

    else:

        ARPESData = np.asarray(pd.read_table(fdir, skiprows=1, header=None))

        return ARPESData


def txtlocate(ffolder, keytext):
    """
    Locate specific txt files containing experimental parameters
    """

    txtfiles = g.glob(ffolder + r'\*.txt')
    for ind, fname in enumerate(txtfiles):
        if keytext in fname:
            txtfile = txtfiles[ind]

    return txtfile


def mat2im(datamat, dtype='uint8', scaling=['normal'], savename=None):
    """
    Convert data matrix to image
    """

    dataconv = np.abs(np.asarray(datamat))
    for scstr in scaling:
        if 'gamma' in scstr:
            gfactors = re.split('gamma|-', scstr)[1:]
            gfactors = u.numFormatConversion(gfactors, form='float', length=2)
            dataconv = gfactors[0]*(dataconv**gfactors[1])

    if 'normal' in scaling:
        dataconv = (255 / dataconv.max()) * (dataconv - dataconv.min())
    elif 'inv' in scaling and 'normal' not in scaling:
        dataconv = 255 - (255 / dataconv.max()) * (dataconv - dataconv.min())

    if dtype == 'uint8':
        imrsc = dataconv.astype(np.uint8)
    im = pim.fromarray(imrsc)

    if savename:
        im.save(savename)
    return im


def im2mat(fdir):
    """
    Convert image to numpy ndarray.
    """

    mat = np.asarray(pim.open(fdir))
    return mat


def metaReadHDF5(hfile, attributes=[], groups=[]):
    """
    Parse the attribute (i.e. metadata) tree in the input HDF5 file and construct a dictionary of attributes

    :Parameters:
        hfile : HDF5 file instance
            Instance of the `h5py.File` class.
        attributes, groups : list, list | [], []
            List of strings representing the names of the specified attribute/group names.
            When specified as None, the components (all attributes or all groups) are ignored.
            When specified as [], all components (attributes/groups) are included.
            When specified as a list of strings, only the attribute/group names matching the strings are retrieved.
    """

    out = {}
    # Extract the file attributes
    if attributes is not None:
        attrdict = dict(hfile.attrs.items()) # Contains all file attributes
        if len(attributes) > 0:
            attrdict = project(attrdict, attributes)

    out = u.dictmerge(out, attrdict)

    # Extract the group information
    if groups is not None:
        # groups = None will not include any group.
        if len(groups) == 0:
            # group = [] will include all groups.
            groups = list(hfile)

        for g in groups:
            gdata = hfile.get(g)
            out[g] = dict(gdata.attrs)
            out[g]['shape'] = gdata.shape

    return out


class hdf5Reader(File):
    """ HDF5 reader class
    """

    def __init__(self, f_addr, ncores=None, **kwds):

        self.faddress = f_addr
        eventEstimator = kwds.pop('estimator', 'Stream_0') # Dataset representing event length
        self.CHUNK_SIZE = int(kwds.pop('chunksz', 1e6))
        super().__init__(name=self.faddress, mode='r', **kwds)

        self.nEvents = self[eventEstimator].size
        self.groupNames = list(self)
        self.groupAliases = [self.readAttribute(self[gn], 'Name', nullval=gn) for gn in self.groupNames]
        # Initialize the look-up dictionary between group aliases and group names
        self.nameLookupDict = dict(zip(self.groupAliases, self.groupNames))
        self.attributeNames = list(self.attrs)

        if (ncores is None) or (ncores > N_CPU) or (ncores < 0):
            self.ncores = N_CPU
        else:
            self.ncores = int(ncores)

    def getGroupNames(self, wexpr=None, woexpr=None, use_alias=False):
        """ Retrieve group names from the loaded hdf5 file with string filtering.

        :Parameters:
            wexpr : str | None
                Expression in a name to leave in the group name list (w = with).
            woexpr : str | None
                Expression in a name to leave out of the group name list (wo = without).
            use_alias : bool | False
                Specification on the use of alias to replace the variable name.

        :Return:
            filteredGroupNames : list
                List of filtered group names.
        """

        # Gather group aliases, if specified
        if use_alias == True:
            groupNames = self.name2alias(self.groupNames)
        else:
            groupNames = self.groupNames

        # Filter the group names
        if (wexpr is None) and (woexpr is None):
            filteredGroupNames = groupNames
        if wexpr:
            filteredGroupNames = [i for i in groupNames if wexpr in i]
        elif woexpr:
            filteredGroupNames = [i for i in groupNames if woexpr not in i]

        return filteredGroupNames

    def getAttributeNames(self, wexpr=None, woexpr=None):
        """ Retrieve attribute names from the loaded hdf5 file with string filtering.

        :Parameters:
            wexpr : str | None
                Expression in a name to leave in the attribute name list (w = with).
            woexpr : str | None
                Expression in a name to leave out of the attribute name list (wo = without).

        :Return:
            filteredAttrbuteNames : list
                List of filtered attribute names.
        """

        if (wexpr is None) and (woexpr is None):
            filteredAttributeNames = self.attributeNames
        elif wexpr:
            filteredAttributeNames = [i for i in self.attributeNames if wexpr in i]
        elif woexpr:
            filteredAttributeNames = [i for i in self.attributeNames if woexpr not in i]

        return filteredAttributeNames

    @staticmethod
    def readGroup(element, *group, amin=None, amax=None, sliced=True):
        """ Retrieve the content of the group(s) in the loaded hdf5 file.

        :Parameter:
            group : list/tuple
                Collection of group names.
            amin, amax : numeric, numeric | None, None
                Minimum and maximum indice to select from the group (dataset).
            sliced : bool | True
                Perform slicing on the group (dataset), if `True`.

        :Return:
            groupContent : list/tuple
                Collection of values of the corresponding groups.
        """

        ngroup = len(group)
        amin, amax = u.intify(amin, amax)
        groupContent = []

        for g in group:
            try:
                if sliced:
                    groupContent.append(element.get(g)[slice(amin, amax)])
                else:
                    groupContent.append(element.get(g))
            except:
                raise ValueError("Group '"+g+"' doesn't have sufficient length for slicing!")

        if ngroup == 1: # Singleton case
            groupContent = groupContent[0]

        return groupContent

    @staticmethod
    def readAttribute(element, *attribute, nullval='None'):
        """ Retrieve the content of the attribute(s) in the loaded hdf5 file.

        :Parameter:
            attribute : list/tuple
                Collection of attribute names.
            nullval : str | 'None'
                Null value to retrieve as a replacement of NoneType.

        :Return:
            attributeContent : list/tuple
                Collection of values of the corresponding attributes.
        """

        nattr = len(attribute)
        attributeContent = []

        for ab in attribute:
            try:
                attributeContent.append(element.attrs[ab].decode('utf-8'))
            except AttributeError: # No need to decode
                attributeContent.append(element.attrs[ab])
            except KeyError: # No such an attribute
                attributeContent.append(nullval)

        if nattr == 1:
            attributeContent = attributeContent[0]

        return attributeContent

    def name2alias(self, names_to_convert):
        """ Find corresponding aliases of the named groups.

        :Parameter:
            names_to_convert : list/tuple
                Names to convert to aliases.

        :Return:
            aliases : list/tuple
                Aliases corresponding to the names.
        """

        aliases = [self.readAttribute(self[ntc], 'Name', nullval=ntc) for ntc in names_to_convert]

        return aliases

    def _assembleGroups(self, gnames, amin=None, amax=None, use_alias=True, dtyp='float32', ret='array'):
        """ Assemble the content values of the selected groups.

        :Parameters:
            gnames : list
                List of group names.
            amin, amax : numeric, numeric | None, None
                Index selection range for all groups.
            use_alias : bool | True
                See `hdf5Reader.getGroupNames()`.
            dtype : str | 'float32'
                Data type string.
            ret : str | 'array'
                Return type specification ('array' or 'dict').
        """

        gdict = {}

        # Add groups to dictionary
        for ign, gn in enumerate(gnames):

            g_dataset = self.readGroup(self, gn, sliced=False)
            g_values = g_dataset[slice(amin, amax)]
            if bool(dtyp):
                g_values = g_values.astype(dtyp)

            # Use the group alias as the dictionary key
            if use_alias == True:
                g_name = self.readAttribute(g_dataset, 'Name', nullval=gn)
                gdict[g_name] = g_values
            # Use the group name as the dictionary key
            else:
                gdict[gn] = g_values

            # print('{}: {}'.format(g_name, g_values.dtype))
        if ret == 'array':
            return np.asarray(list(gdict.values()))
        elif ret == 'dict':
            return gdict

    def summarize(self, form='text', use_alias=True, ret=False, **kwds):
        """
        Summarize the content of the hdf5 file (names of the groups,
        attributes and the selected contents. Output in various user-specified formats.)

        :Parameters:
            form : str | 'text'
                Format to summarize the content of the file into. The options include the following,
                :'dataframe': HDF5 content summarized into a dask dataframe.
                :'dict': HDF5 content (both data and metadata) summarized into a dictionary.
                :'metadict': HDF5 metadata summarized into a dictionary.
                :'text': descriptive text summarizing the HDF5 content.
            use_alias : bool | True
                Specify whether to use the alias to rename the groups.
            ret : bool | False
                Specify whether function return is sought.
            **kwds : keyword arguments

        :Return:
            hdfdict : dict
                Dictionary including both the attributes and the groups,
                using their names as the keys.
            edf : dataframe
                Dataframe (edf = electron dataframe) constructed using only the group values,
                and the column names are the corresponding group names (or aliases).
        """

        # Summarize file information as printed text
        if form == 'text':
            # Print-out header
            print('*** HDF5 file info ***\n',
                    'File address = ' + self.faddress + '\n')

            # Output info on attributes
            print('\n>>> Attributes <<<\n')
            for an in self.attributeNames:
                print(an + ' = {}'.format(self.readAttribute(self, an)))

            # Output info on groups
            print('\n>>> Groups <<<\n')
            for gn in self.groupNames:

                g_dataset = self.readGroup(self, gn, sliced=False)
                g_shape = g_dataset.shape
                g_alias = self.readAttribute(g_dataset, 'Name')

                print(gn + ', Shape = {}, Alias = {}'.format(g_shape, g_alias))

        # Summarize all metadata into a nested dictionary
        elif form == 'metadict':

            # Empty list specifies retrieving all entries, see mpes.metaReadHDF5()
            attributes = kwds.pop('attributes', [])
            groups = kwds.pop('groups', [])

            return metaReadHDF5(self, attributes, groups)

        # Summarize attributes and groups into a dictionary
        elif form == 'dict':

            # Retrieve the range of acquired events
            amin = kwds.pop('amin', None)
            amax = kwds.pop('amax', None)
            amin, amax = u.intify(amin, amax)

            # Output as a dictionary
            # Attribute name stays, stream_x rename as their corresponding attribute name
            # Add groups to dictionary
            hdfdict = self._assembleGroups(self.groupNames, amin=amin, amax=amax,
                            use_alias=use_alias, ret='dict')

            # Add attributes to dictionary
            for an in self.attributeNames:

                hdfdict[an] = self.readAttribute(self, an)

            if ret == True:
                return hdfdict

        # Gather groups into columns of a dataframe
        elif form == 'dataframe':

            self.CHUNK_SIZE = int(kwds.pop('chunksz', 1e6))

            dfParts = []
            chunkSize = min(self.CHUNK_SIZE, self.nEvents / self.ncores)
            nPartitions = int(self.nEvents // chunkSize) + 1
            # Determine the column names
            gNames = kwds.pop('groupnames', self.getGroupNames(wexpr='Stream'))
            colNames = self.name2alias(gNames)

            for p in range(nPartitions): # Generate partitioned dataframe

                # Calculate the starting and ending index of every chunk of events
                eventIDStart = int(p * chunkSize)
                eventIDEnd = int(min(eventIDStart + chunkSize, self.nEvents))
                dfParts.append(d.delayed(self._assembleGroups)(gNames, amin=eventIDStart, amax=eventIDEnd, **kwds))

            # Construct eda (event dask array) and edf (event dask dataframe)
            eda = da.from_array(np.concatenate(d.compute(*dfParts), axis=1).T, chunks=self.CHUNK_SIZE)
            self.edf = ddf.from_dask_array(eda, columns=colNames)

            if ret == True:
                return self.edf

    def convert(self, form, save_addr='./summary', pq_append=False, **kwds):
        """ Format conversion from hdf5 to mat (for Matlab/Python) or ibw (for Igor).

        :Parameters:
            form : str
                The format of the data to convert into.
            save_addr : str | './summary'
                File address to save to.
            pq_append : bool | False
                Option to append to parquet files.
                :True: Append to existing parquet files.
                :False: The existing parquet files will be deleted before new file creation.
        """

        save_fname = u.appendformat(save_addr, form)

        if form == 'mat': # Save dictionary as mat file
            hdfdict = self.summarize(form='dict', ret=True, **kwds)
            sio.savemat(save_fname, hdfdict)

        elif form == 'parquet': # Save dataframe as parquet file
            compression = kwds.pop('compression', 'UNCOMPRESSED')
            engine = kwds.pop('engine', 'fastparquet')

            self.summarize(form='dataframe', **kwds)
            self.edf.to_parquet(save_addr, engine=engine, compression=compression,
                                append=pq_append, ignore_divisions=True)

        elif form == 'ibw':
        # TODO: Save in igor ibw format
            raise NotImplementedError

        else:
            raise NotImplementedError


def saveDict(processor, dictname, form='h5', save_addr='./histogram', **kwds):
    """ Save the binning result dictionary, including the histogram and the
    axes values (edges or midpoints).

    :Parameters:
        processor : class
            Class including all attributes.
        dictname : str
            Namestring of the dictionary to save (such as the attribute name in a class).
        form : str | 'h5'
            Save format, supporting 'mat', 'h5'/'hdf5', 'tiff' (need tifffile) or 'png' (need imageio).
        save_addr : str | './histogram'
            File path to save the binning result.
        **kwds : keyword arguments
            ================  ===========  ===========  ========================================
             keyword           data type     default     meaning
            ================  ===========  ===========  ========================================
              dtyp              string      'float32'    Data type of the histogram
             cutaxis             int            3        The index of axis to cut the 4D data
            slicename           string         'V'       The shared namestring for the 3D slice
            otheraxes            dict         None       Values along other or converted axes
            mat_compression      bool        False       Matlab file compression
            ================  ===========  ===========  ========================================
    """

    dtyp = kwds.pop('dtyp', 'float32')
    sln = kwds.pop('slicename', 'V') # sln = slicename
    save_addr = u.appendformat(save_addr, form)
    otheraxes = kwds.pop('otheraxes', None)

    histdict = getattr(processor, dictname)
    # Include other axes values in the binning dictionary
    if otheraxes:
        histdict = u.dictmerge(histdict, otheraxes)

    if form == 'mat': # Save as mat file (for Matlab)

        do_compression = kwds.pop('mat_compression', False)
        sio.savemat(save_addr, histdict, **kwds)

    elif form in ('h5', 'hdf5'): # Save as hdf5 file

        cutaxis = kwds.pop('cutaxis', 3)
        # Change the bit length of data
        if dtyp not in ('float64', 'float'):
            for dk, dv in histdict.items():
                try:
                    histdict[dk] = dv.astype(dtyp)
                except:
                    pass

        # Save the binned data
        # Save 1-3D data as single datasets
        try:
            hdf = File(save_addr, 'w')
            if processor.nbinaxes < 4:
                hdf.create_dataset('binned/'+sln, data=histdict['binned'])
            # Save 4D data as a list of separated 3D datasets
            elif processor.nbinaxes == 4:
                nddata = np.rollaxis(histdict['binned'], cutaxis)
                n = nddata.shape[0]
                for i in range(n):
                    hdf.create_dataset('binned/'+sln+str(i), data=nddata[i,...])
            else:
                raise NotImplementedError('The output format is undefined for data\
                with higher than four dimensions!')

            # Save the axes in the same group
            for bax in processor.binaxes:
                hdf.create_dataset('axes/'+bax, data=histdict[bax])

        finally:
            hdf.close()

    elif form == 'tiff': # Save as tiff stack

        try:
            import tifffile as ti
            ti.imsave(save_addr, data=histdict['binned'].astype(dtyp))
        except ImportError:
            raise ImportError('tifffile package is not installed locally!')

    elif form == 'png': # Save as png for slices

        import imageio as imio
        cutaxis = kwds.pop('cutaxis', 2)

        if processor.nbaxes == 2:
            imio.imwrite(save_addr[:-3]+'.png', histdict['binned'], format='png')
        if processor.nbinaxes == 3:
            nddata = np.rollaxis(histdict['binned'], cutaxis)
            n = nddata.shape[0]
            for i in range(n):
                wn.simplefilter('ignore', UserWarning)
                imio.imwrite(save_addr[:-3]+'_'+str(i)+'.png', nddata[i,...], format='png')

        elif processor.nbinaxes >= 4:
            raise NotImplementedError('The output format is undefined for data \
            with higher than three dimensions!')

    elif form == 'ibw': # Save as Igor wave

        from igorwriter import IgorWave
        wave = IgorWave(histdict['binned'], name='binned')
        wave.save(save_addr)

    else:
        raise NotImplementedError('Not implemented output format!')


class hdf5Processor(hdf5Reader):
    """ Class for generating multidimensional histogram from hdf5 files.
    """

    def __init__(self, f_addr, **kwds):

        self.faddress = f_addr
        self.ua = kwds.pop('', True)
        self.hdfdict = {}
        self.histdict = {}
        self.axesdict = {}

        super().__init__(f_addr=self.faddress, **kwds)

    def _addBinners(self, axes=None, nbins=None, ranges=None, binDict=None, irregular_bins=False):
        """
        Construct the binning parameters within an instance.
        """

        # Use information specified in binDict, ignore others
        if binDict is not None:
            try:
                self.binaxes = list(binDict['axes'])
                self.nbinaxes = len(self.binaxes)
                self.bincounts = binDict['nbins']
                self.binranges = binDict['ranges']
            except:
                pass # No action when binDict is not specified
        # Use information from other specified parameters if binDict is not given
        else:
            self.binaxes = list(axes)
            self.nbinaxes = len(self.binaxes)

            # Collect the number of bins
            if irregular_bins == False:
                try: # To have the same number of bins on all axes
                    self.bincounts = int(nbins)
                except: # To have different number of bins on each axis
                    self.bincounts = list(map(int, nbins))

            self.binranges = ranges

        # Construct binning steps
        self.binsteps = []
        for bc, (lrange, rrange) in zip(self.bincounts, self.binranges):
            self.binsteps.append((rrange - lrange) / bc)

    @d.delayed
    def _delayedBinning(self, data):
        """
        Lazily evaluated multidimensional binning.

        :Parameters:
            data : numpy array
                Data to bin.

        :Returns:
            hist : numpy array
                Binned histogram.
            edges : list of numpy array
                Bins along each axis of the histogram.
        """

        hist, edges = np.histogramdd(data, bins=self.bincounts, range=self.binranges)

        return hist, edges

    def loadMapping(self):
        """
        Load the mapping parameters
        """

        pass

    def distributedProcessBinning(self, axes=None, nbins=None, ranges=None,
                                binDict=None, chunksz=100000, pbar=True, ret=True, **kwds):
        """
        Compute the photoelectron intensity histogram using dask array.

        :Paramters:
            axes : (list of) strings | None
                Names the axes to bin.
            nbins : (list of) int | None
                Number of bins along each axis.
            ranges : (list of) tuples | None
                Ranges of binning along every axis.
            binDict : dict | None
                Dictionary with specifications of axes, nbins and ranges. If binDict
                is not None. It will override the specifications from other arguments.
            chunksz : numeric (single numeric or tuple)
                Size of the chunk to distribute.
            pbar : bool | True
                Option to display a progress bar.
            ret : bool | True
                :True: returns the dictionary containing binned data explicitly
                :False: no explicit return of the binned data, the dictionary
                generated in the binning is still retained as an instance attribute.

        :Return:
            histdict : dict
                Dictionary containing binned data and the axes values (if `ret = True`).
        """

        # Retrieve the range of acquired events
        amin = kwds.pop('amin', None)
        amax = kwds.pop('amax', None)
        amin, amax = u.intify(amin, amax)

        # Set up the binning parameters
        self._addBinners(axes, nbins, ranges, binDict)

        # Assemble the data to bin in a distributed way
        if (amin is None) and (amax is None):
            dsets = [self[self.nameLookupDict[ax]] for ax in axes]
        else:
            dsets = [self[self.nameLookupDict[ax]][slice(amin, amax)] for ax in axes]

        dsets_distributed = [da.from_array(ds, chunks=(chunksz)) for ds in dsets]
        data_unbinned = da.stack(dsets_distributed, axis=1)
        # if rechunk:
        #     data_unbineed = data_unbinned.rechunk('auto')

        # Compute binned data
        bintask = self._delayedBinning(self, data_unbinned)
        if pbar:
            with ProgressBar():
                self.histdict['binned'], ax_vals = bintask.compute()
        else:
            self.histdict['binned'], ax_vals = bintask.compute()

        for iax, ax in enumerate(axes):
            self.histdict[ax] = ax_vals[iax]

        if ret:
            return self.histdict

    def distributedBinning(self, axes=None, nbins=None, ranges=None, binDict=None,
                            binmethod='lean', pbar=True, ret=True, **kwds):
        """
        :Parameters:
            axes, nbins, ranges, binDict, pbar
                See `mpes.fprocessing.binDataframe()`.
            binmethod : str | 'lean'
                Dataframe binning method ('original' and 'lean').
            ret : bool | True
                :True: returns the dictionary containing binned data explicitly
                :False: no explicit return of the binned data, the dictionary
                generated in the binning is still retained as an instance attribute.
            **kwds : keyword arguments
        """

        # Retrieve the range of acquired events
        amin = kwds.pop('amin', None)
        amax = kwds.pop('amax', None)
        amin, amax = u.intify(amin, amax)

        # Set up the binning parameters
        self._addBinners(axes, nbins, ranges, binDict)
        self.summarize(form='dataframe')
        self.edf = self.edf[amin:amax] # Select event range for binning

        if binmethod == 'original':
            self.histdict = binDataframe(self.edf, ncores=self.ncores, axes=axes, nbins=nbins,
                            ranges=ranges, binDict=binDict, pbar=pbar, **kwds)
        elif binmethod == 'lean':
            self.histdict = binDataframe_lean(self.edf, ncores=self.ncores, axes=axes, nbins=nbins,
                            ranges=ranges, binDict=binDict, pbar=pbar, **kwds)

        if ret:
            return self.histdict

    def localBinning(self, axes=None, nbins=None, ranges=None, binDict=None,
        jittered=False, histcoord='midpoint', ret='dict', **kwds):
        """
        Compute the photoelectron intensity histogram locally after loading all data into RAM.

        :Paramters:
            axes : (list of) strings | None
                Names the axes to bin.
            nbins : (list of) int | None
                Number of bins along each axis.
            ranges : (list of) tuples | None
                Ranges of binning along every axis.
            binDict : dict | None
                Dictionary with specifications of axes, nbins and ranges. If binDict
                is not None. It will override the specifications from other arguments.
            jittered : bool | False
                Determines whether to add jitter to the data to avoid rebinning artefact.
            histcoord : string | 'midpoint'
                The coordinates of the histogram. Specify 'edge' to get the bar edges (every
                dimension has one value more), specify 'midpoint' to get the midpoint of the
                bars (same length as the histogram dimensions).
            ret : bool | True
                :True: returns the dictionary containing binned data explicitly
                :False: no explicit return of the binned data, the dictionary
                generated in the binning is still retained as an instance attribute.
            **kwds : keyword argument
                ================  ==============  ===========  ==========================================
                     keyword         data type      default     meaning
                ================  ==============  ===========  ==========================================
                     amin          numeric/None      None       minimum value of electron sequence
                     amax          numeric/None      None       maximum value of electron sequence
                  jitter_axes          list          axes       list of axes to jitter
                  jitter_bins          list          nbins      list of the number of bins
                jitter_amplitude   numeric/array     0.5        jitter amplitude (single number for all)
                 jitter_ranges         list         ranges      list of the binning ranges
                ================  ==============  ===========  ==========================================

        :Return:
            histdict : dict
                Dictionary containing binned data and the axes values (if `ret = True`).
        """

        # Retrieve the range of acquired events
        amin = kwds.pop('amin', None)
        amax = kwds.pop('amax', None)
        amin, amax = u.intify(amin, amax)

        # Assemble the data for binning, assuming they can be completely loaded into RAM
        self.hdfdict = self.summarize(form='dict', use_alias=self.ua, amin=amin, amax=amax, ret=True)

        # Set up binning parameters
        self._addBinners(axes, nbins, ranges, binDict)

        # Add jitter to the data streams before binning
        if jittered:
            # Retrieve parameters for histogram jittering, the ordering of the jittering
            # parameters is the same as that for the binning
            jitter_axes = kwds.pop('jitter_axes', axes)
            jitter_bins = kwds.pop('jitter_bins', nbins)
            jitter_amplitude = kwds.pop('jitter_amplitude', 0.5*np.ones(len(jitter_axes)))
            jitter_ranges = kwds.pop('jitter_ranges', ranges)

            # Add jitter to the specified dimensions of the data
            for jb, jax, jamp, jr in zip(jitter_bins, jitter_axes, jitter_amplitude, jitter_ranges):

                sz = self.hdfdict[jax].size
                # Calculate the bar size of the histogram in every dimension
                binsize = abs(jr[0] - jr[1])/jb
                self.hdfdict[jax] = self.hdfdict[jax].astype('float32')
                # Jitter as random uniformly distributed noise (W. S. Cleveland)
                self.hdfdict[jax] += jamp * binsize * np.random.uniform(low=-1,
                                        high=1, size=sz).astype('float32')

        # Stack up data from unbinned axes
        data_unbinned = np.stack((self.hdfdict[ax] for ax in axes), axis=1)
        self.hdfdict = {}

        # Compute binned data locally
        self.histdict['binned'], ax_vals = np.histogramdd(data_unbinned, bins=self.bincounts, range=self.binranges)
        del data_unbinned

        for iax, ax in enumerate(axes):
            if histcoord == 'midpoint':
                ax_edge = ax_vals[iax]
                ax_midpoint = (ax_edge[1:] + ax_edge[:-1])/2
                self.histdict[ax] = ax_midpoint
            elif histcoord == 'edge':
                self.histdict[ax] = ax_vals[iax]

        if ret == 'dict':
            return self.histdict
        elif ret == 'histogram':
            histogram = self.histdict.pop('binned')
            self.axesdict = self.histdict.copy()
            self.histdict = {}
            return histogram
        elif ret == False:
            return

    def updateHistogram(self, axes=None, sliceranges=None, ret=False):
        """
        Update the dimensional sizes of the binning results.
        """

        # Input axis order to binning axes order
        binaxes = np.asarray(self.binaxes)
        seqs = [np.where(ax == binaxes)[0][0] for ax in axes]

        for seq, ax, rg in zip(seqs, axes, sliceranges):
            # Update the lengths of binning axes
            seq = np.where(ax == binaxes)[0][0]
            self.histdict[ax] = self.histdict[ax][rg[0]:rg[1]]

            # Update the binned histogram
            tempmat = np.moveaxis(self.histdict['binned'], seq, 0)[rg[0]:rg[1],...]
            self.histdict['binned'] = np.moveaxis(tempmat, 0, seq)

        if ret:
            return self.histdict

    def saveHistogram(self, dictname='histdict', form='h5', save_addr='./histogram', **kwds):
        """
        Save binned histogram and the axes. See `mpes.fprocessing.saveDict()`.
        """

        try:
            saveDict(self, dictname, form, save_addr, **kwds)
        except:
            raise Exception('Saving histogram was unsuccessful!')

    def saveParameters(self, form='h5', save_addr='./binning'):
        """
        Save all the attributes of the binning instance for later use
        (e.g. binning axes, ranges, etc).

        :Parameters:
            form : str | 'h5'
                File format to for saving the parameters ('h5'/'hdf5', 'mat')
            save_addr : str | './binning'
                The address for the to be saved file.
        """

        saveClassAttributes(self, form, save_addr)

    def toSplitter(self):
        """
        Convert to an instance of hdf5Splitter.
        """

        return hdf5Splitter(f_addr=self.faddress)

    def toBandStructure(self):
        """
        Convert to an instance of BandStructure.
        """

        pass


def binPartition(partition, binaxes, nbins, binranges):
    """ Bin the data within a file partition (e.g. dask dataframe).

    :Parameters:
        partition : dataframe partition
            Partition of a dataframe.
        binaxes : list
            List of axes to bin.
        nbins : list
            Number of bins for each binning axis.
        binranges : list
            The range of each axis to bin.

    :Return:
        hist_partition : ndarray
            Histogram from the binning process.
    """

    cols = partition.columns
    binColumns = [cols.get_loc(binax) for binax in binaxes]
    vals = partition.values[:, binColumns]

    hist_partition, _ = np.histogramdd(vals, bins=nbins, range=binranges)

    return hist_partition


def binDataframe(df, ncores=N_CPU, axes=None, nbins=None, ranges=None,
                binDict=None, pbar=True, jittered=True, pbenv='classic', **kwds):
    """
    Calculate multidimensional histogram from columns of a dask dataframe.
    Prof. Yves Acremann's method.

    :Paramters:
        axes : (list of) strings | None
            Names the axes to bin.
        nbins : (list of) int | None
            Number of bins along each axis.
        ranges : (list of) tuples | None
            Ranges of binning along every axis.
        binDict : dict | None
            Dictionary with specifications of axes, nbins and ranges. If binDict
            is not None. It will override the specifications from other arguments.
        pbar : bool | True
            Option to display a progress bar.
        pbenv : str | 'classic'
            Progress bar environment ('classic' for generic version and
            'notebook' for notebook compatible version).
        jittered : bool | True
            Option to add histogram jittering during binning.
        **kwds : keyword arguments
            See `mpes.fprocessing.hdf5Processor.localBinning()`.

    :Return:
        histdict : dict
            Dictionary containing binned data and the axes values (if `ret = True`).
    """

    histdict = {}
    partitionResults = [] # Partition-level results
    tqdm = u.tqdmenv(pbenv)

    # Add jitter to all the partitions before binning
    if jittered:
        # Retrieve parameters for histogram jittering, the ordering of the jittering
        # parameters is the same as that for the binning
        jitter_axes = kwds.pop('jitter_axes', axes)
        jitter_bins = kwds.pop('jitter_bins', nbins)
        jitter_amplitude = kwds.pop('jitter_amplitude', 0.5*np.ones(len(jitter_axes)))
        jitter_ranges = kwds.pop('jitter_ranges', ranges)

        # Add jitter to the specified dimensions of the data
        for jb, jax, jamp, jr in zip(jitter_bins, jitter_axes, jitter_amplitude, jitter_ranges):

            # Calculate the bar size of the histogram in every dimension
            binsize = abs(jr[0] - jr[1])/jb
            # Jitter as random uniformly distributed noise (W. S. Cleveland)
            df[jax] += df.map_partitions(getJitter, amp=jamp*binsize, col=jax)

    # Main loop for binning
    for i in tqdm(range(0, df.npartitions, ncores), disable=not(pbar)):

        coreTasks = [] # Core-level jobs
        for j in range(0, ncores):

            ij = i + j
            if ij >= df.npartitions:
                break

            dfPartition = df.get_partition(ij) # Obtain dataframe partition
            coreTasks.append(d.delayed(binPartition)(dfPartition, axes, nbins, ranges))

        if len(coreTasks) > 0:
            coreResults = d.compute(*coreTasks)

            # Combine all core results for a dataframe partition
            partitionResult = np.zeros_like(coreResults[0])
            for coreResult in coreResults:
                partitionResult += coreResult

            partitionResults.append(partitionResult)
            # del partitionResult

        del coreTasks

    # Combine all partition results
    fullResult = np.zeros_like(partitionResults[0])
    for pr in partitionResults:
        fullResult += np.nan_to_num(pr)

    # Load into dictionary
    histdict['binned'] = fullResult.astype('float32')
    # Calculate axes values
    for iax, ax in enumerate(axes):
        axrange = ranges[iax]
        histdict[ax] = np.linspace(axrange[0], axrange[1], nbins[iax])

    return histdict


def binDataframe_lean(df, ncores=N_CPU, axes=None, nbins=None, ranges=None,
                binDict=None, pbar=True, jittered=True, pbenv='classic', **kwds):
    """
    Calculate multidimensional histogram from columns of a dask dataframe.

    :Paramters:
        axes : (list of) strings | None
            Names the axes to bin.
        nbins : (list of) int | None
            Number of bins along each axis.
        ranges : (list of) tuples | None
            Ranges of binning along every axis.
        binDict : dict | None
            Dictionary with specifications of axes, nbins and ranges. If binDict
            is not None. It will override the specifications from other arguments.
        pbar : bool | True
            Option to display a progress bar.
        pbenv : str | 'classic'
            Progress bar environment ('classic' for generic version and 'notebook' for notebook compatible version).
        jittered : bool | True
            Option to add histogram jittering during binning.
        **kwds : keyword arguments
            See `mpes.fprocessing.hdf5Processor.localBinning()`.

    :Return:
        histdict : dict
            Dictionary containing binned data and the axes values (if `ret = True`).
    """

    histdict = {}
    fullResult = np.zeros(tuple(nbins)) # Partition-level results
    tqdm = u.tqdmenv(pbenv)

    # Add jitter to all the partitions before binning
    if jittered:
        # Retrieve parameters for histogram jittering, the ordering of the jittering
        # parameters is the same as that for the binning
        jitter_axes = kwds.pop('jitter_axes', axes)
        jitter_bins = kwds.pop('jitter_bins', nbins)
        jitter_amplitude = kwds.pop('jitter_amplitude', 0.5*np.ones(len(jitter_axes)))
        jitter_ranges = kwds.pop('jitter_ranges', ranges)

        # Add jitter to the specified dimensions of the data
        for jb, jax, jamp, jr in zip(jitter_bins, jitter_axes, jitter_amplitude, jitter_ranges):

            # Calculate the bar size of the histogram in every dimension
            binsize = abs(jr[0] - jr[1])/jb
            # Jitter as random uniformly distributed noise (W. S. Cleveland)
            df[jax] += df.map_partitions(getJitter, amp=jamp*binsize, col=jax)

    # Main loop for binning
    for i in tqdm(range(0, df.npartitions, ncores), disable=not(pbar)):

        coreTasks = [] # Core-level jobs
        for j in range(0, ncores):

            ij = i + j
            if ij >= df.npartitions:
                break

            dfPartition = df.get_partition(ij) # Obtain dataframe partition
            coreTasks.append(d.delayed(binPartition)(dfPartition, axes, nbins, ranges))

        if len(coreTasks) > 0:
            coreResults = d.compute(*coreTasks)

            # Combine all core results for a dataframe partition
            partitionResult = reduce(_arraysum, coreResults)
            fullResult += partitionResult
            # del partitionResult

        del coreTasks

    # Load into dictionary
    histdict['binned'] = fullResult.astype('float32')
    # Calculate axes values
    for iax, ax in enumerate(axes):
        axrange = ranges[iax]
        histdict[ax] = np.linspace(axrange[0], axrange[1], nbins[iax])

    return histdict


def getJitter(df, amp, col):
    """ Calculate jittering for a dataframe column.

    :Parameters:
        df : dataframe
            Dataframe to add noise/jittering to.
        amp : numeric
            Amplitude scaling for the jittering noise.
        col : str
            Name of the column to add jittering to.

    :Return:
        Uniformly distributed noise vector with specified amplitude and size.
    """

    colsize = df[col].size
    return amp*np.random.uniform(low=-1, high=1, size=colsize)


class hdf5Splitter(hdf5Reader):
    """
    Class to split large hdf5 files.
    """

    def __init__(self, f_addr, **kwds):

        self.faddress = f_addr
        self.splitFilepaths = []
        super().__init__(f_addr=self.faddress, **kwds)

    @d.delayed
    def _split_file(self, idx, save_addr, namestr):
        """ Split file generator.
        """

        evmin, evmax = self.eventList[idx], self.eventList[idx+1]
        fpath = save_addr + namestr + str(idx+1) + '.h5'

        # Use context manager to open hdf5 file
        with File(fpath, 'w') as fsp:

            # Copy the attributes
            for attr, attrval in self.attrs.items():
                fsp.attrs[attr] = attrval

            # Copy the segmented groups and their attributes
            for gp in self.groupNames:
                #self.copy(gn, fsp[gn])
                fsp.create_dataset(gp, data=self.readGroup(self, gp, amin=evmin, amax=evmax))
                for gattr, gattrval in self[gp].attrs.items():
                    fsp[gp].attrs[gattr] = gattrval

        return(fpath)

    def split(self, nsplit, save_addr='./', namestr='split_',
                split_group='Stream_0', pbar=False):
        """
        Split and save an hdf5 file.

        :Parameters:
            nsplit : int
                Number of split files.
            save_addr : str | './'
                Directory to store the split files.
            namestr : str | 'split_'
                Additional namestring attached to the front of the filename.
            split_group : str | 'Stream_0'
                Name of the example group to split for file length reference.
            pbar : bool | False
                Enable (when True)/Disable (when False) the progress bar.
        """

        nsplit = int(nsplit)
        self.splitFilepaths = [] # Refresh the path when re-splitting
        self.eventLen = self[split_group].size
        self.eventList = np.linspace(0, self.eventLen, nsplit+1, dtype='int')
        tasks = []

        # Distributed file splitting
        for isp in range(nsplit):

            tasks.append(self._split_file(isp, save_addr, namestr))

        if pbar:
            with ProgressBar():
                self.splitFilepaths = d.compute(*tasks)
        else:
            self.splitFilepaths = d.compute(*tasks)

    def subset(self, file_id):
        """
        Spawn an instance of hdf5Processor from a specified split file.
        """

        if self.splitFilepaths:
            return hdf5Processor(f_addr=self.splitFilepaths[file_id])

        else:
            raise ValueError("No split files are present.")

    def toProcessor(self):
        """
        Change to an hdf5Processor instance.
        """

        return hdf5Processor(f_addr=self.faddress)


def readDataframe(folder=None, files=None, ftype='parquet', **kwds):
    """ Read stored files from a folder into a dataframe.

    :Parameters:
        folder, files : str, list/tuple | None, None
            Folder path of the files or a list of file paths. The folder path has
            the priority such that if it's specified, the specified files will be ignored.
        ftype : str | 'parquet'
            File type to read ('h5' or 'hdf5', 'parquet', 'json', 'csv', etc).
            If a folder path is given, all files of the specified type are read
            into the dataframe in the reading order.
        **kwds : keyword arguments
            See the keyword arguments for the specific file parser in `dask.dataframe` module.

    :Return:
        Dask dataframe read from specified files.
    """

    # ff (folder or files) is a folder or a list/tuple of files
    if folder is not None:
        ff = folder
        files = g.glob(folder + '/*.' + ftype)

    elif folder is None:
        if files is not None:
            ff = files # List of file paths
        else:
            raise ValueError('Either the folder or file path should be provided!')

    if ftype == 'parquet':
        return ddf.read_parquet(ff, **kwds)

    elif ftype in ('h5', 'hdf5'):
        return ddf.multi.concat_indexed_dataframes([hdf5Processor(f).
                summarize('dataframe', ret=True, **kwds) for f in files])

    elif ftype == 'json':
        return ddf.read_json(ff, **kwds)

    elif ftype == 'csv':
        return ddf.read_csv(ff, **kwds)

    else:
        try:
            return ddf.read_table(ff, **kwds)
        except:
            raise Exception('The file format cannot be understood!')


class dataframeProcessor(MapParser):
    """
    Processs the parquet file converted from single events data.
    """

    def __init__(self, datafolder, paramfolder='', datafiles=[], ncores=None):

        self.datafolder = datafolder
        self.paramfolder = paramfolder
        self.datafiles = datafiles
        self.histogram = None
        self.histdict = {}
        self.npart = 0

        # Instantiate the MapParser class (contains parameters related to binning and image transformation)
        super().__init__(file_sorting=False, folder=paramfolder)

        if (ncores is None) or (ncores > N_CPU) or (ncores < 0):
            self.ncores = N_CPU
        else:
            self.ncores = int(ncores)

    @property
    def nrow(self):
        """ Number of rows in the distributed dataframe.
        """

        return len(self.edf.index)

    @property
    def ncol(self):
        """ Number of columns in the distrbuted dataframe.
        """

        return len(self.edf.columns)

    def read(self, source='folder', ftype='parquet', fids=[], update='', **kwds):
        """ Read into distributed dataframe.

        :Parameters:
            source : str | 'folder'
                Source of the file readout.
                :'folder': Read from the provided data folder.
                :'files': Read from the provided list of file addresses.
            ftype : str | 'parquet'
                Type of file to read into dataframe ('h5' or 'hdf5', 'parquet', 'json', 'csv').
            fids : list | []
                IDs of the files to be selected (see `mpes.base.FileCollection.select()`).
                Specify 'all' to read all files of the given file type.
            update : str | ''
                File selection update option (see `mpes.base.FileCollection.select()`).
            **kwds : keyword arguments
                See keyword arguments in `mpes.readDataframe()`.
        """

        # Create the single-event dataframe
        if source == 'folder':
            self.edf = readDataframe(folder=self.datafolder, files=[], ftype=ftype, **kwds)

        elif source == 'files':
            if len(self.datafiles) > 0: # When filenames are specified
                self.edf = readDataframe(folder=None, files=self.datafiles, ftype=ftype, **kwds)
            else:
                # When only the datafolder address is given but needs to read partial files,
                # first gather files from the folder, then select files and read into dataframe
                self.gather(folder=self.datafolder, identifier=r'/*.'+ftype, file_sorting=True)

                if len(fids) == 0:
                    print('Nothing is read since no file IDs (fids) is specified!')
                    self.datafiles = self.select(ids=fids, update='', ret='selected')
                elif fids == 'all':
                    self.datafiles = self.select(ids=list(range(len(self.files))), update='', ret='selected')
                else:
                    self.datafiles = self.select(ids=fids, update='', ret='selected')

                self.edf = readDataframe(files=self.datafiles, ftype=ftype, **kwds)

        self.npart = self.edf.npartitions

    def _addBinners(self, axes=None, nbins=None, ranges=None, binDict=None):
        """ Construct the binning parameters within an instance.
        """

        # Use information specified in binDict, ignore others
        if binDict is not None:
            try:
                self.binaxes = list(binDict['axes'])
                self.nbinaxes = len(self.binaxes)
                self.bincounts = binDict['nbins']
                self.binranges = binDict['ranges']
            except:
                pass # No action when binDict is not specified
        # Use information from other specified parameters if binDict is not given
        else:
            self.binaxes = list(axes)
            self.nbinaxes = len(self.binaxes)

            # Collect the number of bins
            try: # To have the same number of bins on all axes
                self.bincounts = int(nbins)
            except: # To have different number of bins on each axis
                self.bincounts = list(map(int, nbins))

            self.binranges = ranges

        # Construct binning steps
        self.binsteps = []
        for bc, (lrange, rrange) in zip(self.bincounts, self.binranges):
            self.binsteps.append((rrange - lrange) / bc)

    # Column operations
    def appendColumn(self, colnames, colvals):
        """ Append columns to dataframe.

        :Parameters:
            colnames : list/tuple
                New column names.
            colvals : numpy array/list
                Entries of the new columns.
        """

        colnames = list(colnames)
        colvals = [colvals]
        ncn = len(colnames)
        ncv = len(colvals)

        if ncn != ncv:
            errmsg = 'The names and values of the columns need to have the same dimensions.'
            raise ValueError(errmsg)

        else:
            for cn, cv in zip(colnames, colvals):
                self.edf = self.edf.assign(**{cn:ddf.from_array(cv)})

    def deleteColumn(self, colnames):
        """ Delete columns

        :Parameters:
            colnames : str/list/tuple
                List of column names to be dropped.
        """

        self.edf = self.edf.drop(colnames, axis=1)

    def applyFilter(self, colname, lb=-np.inf, ub=np.inf, update='replace', ret=False):
        """ Application of bound filters to a specified column (can be used consecutively).

        :Parameters:
            colname : str
                Name of the column to filter.
            lb, ub : numeric, numeric | -infinity, infinity
                The lower and upper bounds used in the filtering.
            update : str | 'replace'
                Update option for the filtered dataframe.
            ret : bool | False
                Return option for the filtered dataframe.
        """

        if ret == True:
            return self.edf[(self.edf[colname] > lb) & (self.edf[colname] < ub)]

        if update == 'replace':
            self.edf = self.edf[(self.edf[colname] > lb) & (self.edf[colname] < ub)]

    def columnApply(self, mapping, rescolname, **kwds):
        """ Apply a user-defined function (e.g. partial function) to an existing column.

        :Parameters:
            mapping : function
                Function to apply to the column.
            rescolname : str
                Name of the resulting column.
            **kwds : keyword arguments
                Keyword arguments of the user-input mapping function.
        """

        self.edf[rescolname] = mapping(**kwds)

    def transformColumn(self, oldcolname, mapping, newcolname='Transformed', args=(), update='append', **kwds):
        """ Apply a simple function to an existing column.

        :Parameters:
            oldcolname : str
                The name of the column to use for computation.
            mapping : function
                Functional map to apply to the values of the old column.
            newcolname : str | 'Transformed'
                New column name to be added to the dataframe.
            args : tuple | ()
                Additional arguments of the functional map.
            update : str | 'append'
                Updating option.
                'append' = append to the current dask dataframe as a new column with the new column name.
                'replace' = replace the values of the old column.
            **kwds : keyword arguments
                Additional arguments for the `dask.dataframe.apply()` function.
        """

        if update == 'append':
            self.edf[newcolname] = self.edf[oldcolname].apply(mapping, args=args, meta=('x', 'f8'), **kwds)
        elif update == 'replace':
            self.edf[oldcolname] = self.edf[oldcolname].apply(mapping, args=args, meta=('x', 'f8'), **kwds)

    def transformColumn2D(self, map2D, X, Y, **kwds):
        """ Apply a mapping simultaneously to two dimensions.

        :Parameters:
            map2D : function
                2D mapping function.
            X, Y : series, series
                The two columns of the dataframe to apply mapping to.
            **kwds : keyword arguments
                Additional arguments for the 2D mapping function.
        """

        newX = kwds.pop('newX', X)
        newY = kwds.pop('newY', Y)

        self.edf[newX], self.edf[newY] = map2D(self.edf[X], self.edf[Y], **kwds)

    def applyKCorrection(self, X='X', Y='Y', newX='Xm', newY='Ym', **kwds):
        """ Calculate and replace the X and Y values with their distortion-correction version.
        This method can be reused.

        :Parameters:
            X, Y : str, str | 'X', 'Y'
                Labels of the columns before momentum distortion correction.
            newX, newY : str, str | 'Xm', 'Ym'
                Labels of the columns after momentum distortion correction.
        """

        self.transformColumn2D(map2D=self.wMap, X=X, Y=Y, newX=newX, newY=newY, **kwds)

    def appendKAxis(self, x0, y0, X='X', Y='Y', newX='kx', newY='ky', **kwds):
        """ Calculate and append the k axis coordinates (kx, ky) to the events dataframe.
        This method can be reused.
        """

        self.transformColumn2D(map2D=self.kMap, X=X, Y=Y, newX=newX, newY=newY, r0=x0, c0=y0, **kwds)

    def appendEAxis(self, E0, **kwds):
        """ Calculate and append the E axis to the events dataframe.
        This method can be reused.

        :Parameter:
            E0 : numeric
                Time-of-flight offset.
        """

        self.columnApply(mapping=self.EMap, rescolname='E', E0=E0, **kwds)

    # Row operation
    def appendRow(self, folder=None, df=None, ftype='parquet', **kwds):
        """ Append rows read from other files to existing dataframe.

        :Parameters:
            folder : str | None
                Folder directory for the files to append to the existing dataframe
                (i.e. when appending parquet files).
            df : dataframe | None
                Dataframe to append to the exisitng dataframe.
            ftype : str | 'parquet'
                File type ('parquet', 'dataframe')
            **kwds : keyword arguments
                Additional arguments to submit to `dask.dataframe.append()`.
        """

        if ftype == 'parquet':
            return self.edf.append(self.read(folder), **kwds)
        elif ftype == 'dataframe':
            return self.edf.append(df, **kwds)
        else:
            raise NotImplementedError

    # Complex operation
    def distributedBinning(self, axes, nbins, ranges, binDict=None, pbar=True,
                            binmethod='lean', ret=False, **kwds):
        """ Binning the dataframe to a multidimensional histogram.

        :Parameters:
            axes, nbins, ranges, binDict, pbar
                See `mpes.fprocessing.binDataframe()`.
            binmethod : str | 'lean'
                Dataframe binning method ('original' and 'lean').
            ret : bool | False
                Option to return binning results as a dictionary.
        """

        # Set up the binning parameters
        self._addBinners(axes, nbins, ranges, binDict)
        edf = kwds.pop('df', self.edf)
        #self.edf = self.edf[amin:amax] # Select event range for binning

        self.histdict = {}
        if binmethod == 'original':
            self.histdict = binDataframe(self.edf, ncores=self.ncores, axes=axes, nbins=nbins,
                            ranges=ranges, binDict=binDict, pbar=pbar, **kwds)
        elif binmethod == 'lean':
            self.histdict = binDataframe_lean(self.edf, ncores=self.ncores, axes=axes, nbins=nbins,
                            ranges=ranges, binDict=binDict, pbar=pbar, **kwds)

        if ret:
            return self.histdict

    def convert(self, form='parquet', save_addr=None, namestr='/data', pq_append=False, **kwds):
        """ Update or convert to other file formats.

        :Parameters:
            form : str | 'parquet'
                File format to convert into.
            save_addr : str | None
                Path of the folder to save the converted files to.
            namestr : '/data'
                Extra namestring attached to the filename.
            pq_append : bool | False
                Option to append to the existing parquet file (if `True`) in the specified folder,
                otherwise the existing parquet files will be deleted before writing new files in.
            **kwds : keyword arguments
                See extra keyword arguments in `dask.dataframe.to_parquet()` for parquet conversion,
                or in `dask.dataframe.to_hdf()` for HDF5 conversion.
        """

        if form == 'parquet':
            compression = kwds.pop('compression', 'UNCOMPRESSED')
            engine = kwds.pop('engine', 'fastparquet')
            self.edf.to_parquet(save_addr, engine=engine, compression=compression,
                                append=pq_append, ignore_divisions=True, **kwds)

        elif form in ('h5', 'hdf5'):
            self.edf.to_hdf(save_addr, namestr, **kwds)

        elif form == 'json':
            self.edf.to_json(save_addr, **kwds)

    def saveHistogram(self, form, save_addr, dictname='histdict', **kwds):
        """ Export binned histogram in other formats.

        :Parameters:
            See `mpes.fprocessing.saveDict()`.
        """

        try:
            saveDict(self, dictname, form, save_addr, **kwds)
        except:
            raise Exception('Saving histogram was unsuccessful!')

    def toDataStructure(self):
        """ Convert to the xarray data structure from existing binned data.

        :Return:
            An instance of `BandStructure` or `MPESDataset` from the `mpes.bandstructure` module.
        """

        if bool(self.histdict):
            coords = project(self.histdict, self.binaxes)

            if self.nbinaxes == 3:
                return bs.BandStructure(data=self.histdict['binned'],
                        coords=coords, dims=self.binaxes, datakey='')
            elif self.nbinaxes > 3:
                return bs.MPESDataset(data=self.histdict['binned'],
                        coords=coords, dims=self.binaxes, datakey='')

        else:
            raise ValueError('No binning results are available!')


class parquetProcessor(dataframeProcessor):
    """
    Legacy version of the `mpes.fprocessing.dataframeProcessor` class.
    """

    def __init__(self, folder, files=[], source='folder', ftype='parquet',
                fids=[], update='', ncores=None, **kwds):

        super().__init__(datafolder=folder, paramfolder=folder, datafiles=files, ncores=ncores)
        self.folder = folder
        # Read only the parquet files from the given folder/files
        self.read(source=source, ftype=ftype, fids=fids, update=update, **kwds)
        self.npart = self.edf.npartitions


def _arraysum(array_a, array_b):
    """
    Calculate the sum of two arrays.
    """

    return array_a + array_b


class parallelHDF5Processor(FileCollection):
    """
    Class for parallel processing of hdf5 files.
    """

    def __init__(self, files=[], file_sorting=True, folder=None):

        super().__init__(files=files, file_sorting=file_sorting, folder=folder)
        self.metadict = {}
        self.results = {}
        self.combinedresult = {}

    def _parse_metadata(self, attributes, groups):
        """
        Parse the metadata from all HDF5 files.

        :Parameters:
            attributes, groups : list, list
                See `mpes.fprocessing.metaReadHDF5()`.
        """

        for fid in range(self.nfiles):
            output = self.subset(fid).summarize(form='metadict', attributes=attributes, groups=groups)
            self.metadict = u.dictmerge(self.metadict)

    def subset(self, file_id):
        """
        Spawn an instance of hdf5Processor from a specified substituent file.

        :Parameter:
            file_id : int
                Integer-numbered file ID (from 0 to self.nfiles).
        """

        if self.files:
            return hdf5Processor(f_addr=self.files[int(file_id)])

        else:
            raise ValueError("No substituent file is present.")

    def summarize(self, form='dataframe', ret=False, **kwds):
        """
        Summarize the measurement information from all HDF5 files.

        :Parameters:
            form : str | 'dataframe'
                Format of the files to summarize into.
            ret : bool | False
                Specification on value return.
            **kwds : keyword arguments
                See keyword arguments in `mpes.fprocessing.readDataframe()`.
        """

        if form == 'text':

            raise NotImplementedError

        elif form == 'metadict':

            self.metadict = {}

            if ret == True:
                return self.metadict

        elif form == 'dataframe':
            self.edfhdf = readDataframe(**kwds)

            if ret == True:
                return self.edfhdf

    def parallelBinning(self, axes, nbins, ranges, scheduler='threads', combine=True,
    histcoord='midpoint', pbar=True, binning_kwds={}, compute_kwds={}, ret=False):
        """
        Parallel computation of the multidimensional histogram from file segments.

        :Parameters:
            axes : (list of) strings | None
                Names the axes to bin.
            nbins : (list of) int | None
                Number of bins along each axis.
            ranges : (list of) tuples | None
                Ranges of binning along every axis.
            scheduler : str | 'threads'
                Type of distributed scheduler ('threads', 'processes', 'synchronous')
            combine : bool | True
                Option to combine the results obtained from distributed binning.
            histcoord : string | 'midpoint'
                The coordinates of the histogram. Specify 'edge' to get the bar edges (every
                dimension has one value more), specify 'midpoint' to get the midpoint of the
                bars (same length as the histogram dimensions).
            pbar : bool | true
                Option to display the progress bar.
            binning_kwds : dict | {}
                Keyword arguments to be included in `mpes.fprocessing.hdf5Processor.localBinning()`.
            compute_kwds : dict | {}
                Keyword arguments to specify in `dask.compute()`.
        """

        binTasks = []
        self.binaxes = axes
        self.nbinaxes = len(axes)
        self.bincounts = nbins
        self.binranges = ranges

        # Construct binning steps
        self.binsteps = []
        for bc, (lrange, rrange) in zip(self.bincounts, self.binranges):
            self.binsteps.append((rrange - lrange) / bc)

        # Reset containers of results
        self.results = {}
        self.combinedresult = {}

        # Execute binning tasks
        if combine == True: # Combine results in the process of binning
            binning_kwds = u.dictmerge({'ret':'histogram'}, binning_kwds)
            # Construct binning tasks
            for f in self.files:
                binTasks.append(d.delayed(hdf5Processor(f).localBinning)
                                (axes=axes, nbins=nbins, ranges=ranges, **binning_kwds))
            if pbar:
                with ProgressBar():
                    self.combinedresult['binned'] = reduce(_arraysum,
                    d.compute(*binTasks, scheduler=scheduler, **compute_kwds))
            else:
                self.combinedresult['binned'] = reduce(_arraysum,
                d.compute(*binTasks, scheduler=scheduler, **compute_kwds))

            del binTasks

            # Calculate and store values of the axes
            for iax, ax in enumerate(self.binaxes):
                p_start, p_end = self.binranges[iax]
                self.combinedresult[ax] = u.calcax(p_start, p_end, self.bincounts[iax], ret=histcoord)

            if ret:
                return self.combinedresult

        else: # Return all task outcome of binning (not recommended due to the size)
            for f in self.files:
                binTasks.append(d.delayed(hdf5Processor(f).localBinning)
                                (axes=axes, nbins=nbins, ranges=ranges, **binning_kwds))
            if pbar:
                with ProgressBar():
                    self.results = d.compute(*binTasks, scheduler=scheduler, **compute_kwds)
            else:
                self.results = d.compute(*binTasks, scheduler=scheduler, **compute_kwds)

            del binTasks

            if ret:
                return self.results

    def combineResults(self, ret=True):
        """
        Combine the results from all segments (only when self.results is non-empty).

        :Parameters:
            ret : bool | True
                :True: returns the dictionary containing binned data explicitly
                :False: no explicit return of the binned data, the dictionary
                generated in the binning is still retained as an instance attribute.

        :Return:
            combinedresult : dict
                Return combined result dictionary (if `ret == True`).
        """

        try:
            binnedhist = np.stack([self.results[i]['binned'] for i in range(self.nfiles)], axis=0).sum(axis=0)

            # Transfer the results to combined result
            self.combinedresult = self.results[0].copy()
            self.combinedresult['binned'] = binnedhist
        except:
            pass

        if ret:
            return self.combinedresult

    def convert(self, form='parquet', save_addr='./summary', append_to_folder=False,
                pbar=True, pbenv='classic', **kwds):
        """
        Convert files to another format (e.g. parquet).

        :Parameters:
            form : str | 'parquet'
                File format to convert into.
            save_addr : str | './summary'
                Path of the folder for saving parquet files.
            append_to_folder : bool | False
                Option to append to the existing parquet files in the specified folder,
                otherwise the existing parquet files will be deleted first. The HDF5 files
                in the same folder are kept intact.
            pbar : bool | True
                Option to display progress bar.
            pbenv : str | 'classic'
                Specification of the progress bar environment ('classic' for generic version
                and 'notebook' for notebook compatible version).
            **kwds : keyword arguments
                See `mpes.fprocessing.hdf5Processor.convert()`.
        """

        tqdm = u.tqdmenv(pbenv)

        if os.path.exists(save_addr) and os.path.isdir(save_addr):
            # In an existing folder, clean up the files if specified
            existing_files = g.glob(save_addr + r'/*')
            n_existing_files = len(existing_files)

            # Remove existing files in the folder before writing into it
            if (n_existing_files > 0) and (append_to_folder == False):
                for f_exist in existing_files:
                    if '.h5' not in f_exist: # Keep the calibration files
                        os.remove(f_exist)

        for fi in tqdm(range(self.nfiles), disable=not(pbar)):
            subproc = self.subset(file_id=fi)
            subproc.convert(form=form, save_addr=save_addr, pq_append=True, **kwds)

    def updateHistogram(self, axes=None, sliceranges=None, ret=False):
        """
        Update the dimensional sizes of the binning results.

        :Parameters:
            axes : tuple/list | None
                Collection of the names of axes for size change.
            sliceranges : tuple/list/array | None
                Collection of ranges, e.g. (start_position, stop_position) pairs,
                for each axis to be updated.
            ret : bool | False
                Option to return updated histogram.
        """

        # Input axis order to binning axes order
        binaxes = np.asarray(self.binaxes)
        seqs = [np.where(ax == binaxes)[0][0] for ax in axes]

        for seq, ax, rg in zip(seqs, axes, sliceranges):
            # Update the lengths of binning axes
            self.combinedresult[ax] = self.combinedresult[ax][rg[0]:rg[1]]

            # Update the binned histogram
            tempmat = np.moveaxis(self.combinedresult['binned'], seq, 0)[rg[0]:rg[1],...]
            self.combinedresult['binned'] = np.moveaxis(tempmat, 0, seq)

        if ret:
            return self.combinedresult

    def saveHistogram(self, dictname='combinedresult', form='h5', save_addr='./histogram', **kwds):
        """
        Save binned histogram and the axes.

        :Parameters:
            See `mpes.fprocessing.saveDict()`.
        """

        try:
            saveDict(self, dictname, form, save_addr, **kwds)
        except:
            raise Exception('Saving histogram was unsuccessful!')

    def saveParameters(self, form='h5', save_addr='./binning'):
        """
        Save all the attributes of the binning instance for later use
        (e.g. binning axes, ranges, etc).

        :Parameters:
            form : str | 'h5'
                File format to for saving the parameters ('h5'/'hdf5', 'mat').
            save_addr : str | './binning'
                The address for the to be saved file.
        """

        saveClassAttributes(self, form, save_addr)


def extractEDC(folder=None, files=[], axes=['t'], bins=[1000], ranges=[(65000, 100000)],
                binning_kwds={'jittered':True}, ret=True, **kwds):
    """ Extract EDCs from a list of bias scan files.
    """

    pp = parallelHDF5Processor(folder=folder, files=files)
    if len(files) == 0:
        pp.gather(identifier='/*.h5')
    pp.parallelBinning(axes=axes, nbins=bins, ranges=ranges, combine=False, ret=False,
                        binning_kwds=binning_kwds, **kwds)

    edcs = [pp.results[i]['binned'] for i in range(len(pp.results))]
    tof = pp.results[0][axes[0]]
    traces = np.asarray(edcs)
    del pp

    if ret:
        return traces, tof


def readBinnedhdf5(fpath, combined=True, typ='float32'):
    """
    Read binned hdf5 file (3D/4D data) into a dictionary.

    :Parameters:
        fpath : str
            File path
        combined : bool | True
            Specify if the volume slices are combined.
        typ : str | 'float32'
            Data type of the numerical values in the output dictionary

    :Return:
        out : dict
            Dictionary with keys being the axes and the volume (slices).
    """

    f = File(fpath, 'r')
    out = {}

    # Read the axes group
    for ax, axval in f['axes'].items():
        out[ax] = axval[...]

    # Read the binned group
    group = f['binned']
    itemkeys = group.keys()
    nbinned = len(itemkeys)

    # Binned 3D matrix
    if (nbinned == 1) or (combined == False):
        for ik in itemkeys:
            out[ik] = np.asarray(group[ik], dtype=typ)

    # Binned 4D matrix
    elif (nbinned > 1) or (combined == True):
        val = []
        itemkeys_sorted = nts.natsorted(itemkeys)
        for ik in itemkeys_sorted:
            val.append(group[ik])
        out['V'] = np.asarray(val, dtype=typ)

    return out


# =================== #
# Data transformation #
# =================== #

def fftfilter2d(datamat):

    r, c = datamat.shape
    x, y = np.meshgrid(np.arange(-r / 2, r / 2), np.arange(-c / 2, c / 2))
    zm = np.zeros_like(datamat.T)

    ftmat = (nft.fftshift(nft.fft2(datamat))).T

    # Construct peak center coordinates array using rotation
    x0, y0 = -80, -108
    # Conversion factor for radius (half-width half-maximum) of Gaussian
    rgaus = 2 * np.log(2)
    sx, sy = 10 / rgaus, 10 * (c / r) / rgaus
    alf, bet = np.arctan(r / c), np.arctan(c / r)
    rotarray = np.array([0, 2 * alf, 2 * (alf + bet), -2 * bet])
    xy = [np.dot(rot2d(roth, 'rad'), np.array([x0, y0])) for roth in rotarray]

    # Generate intermediate positions and append to peak center coordinates
    # array
    for everynumber in range(4):
        n = everynumber % 4
        xy.append((xy[n] + xy[n - 1]) / 2)

    # Construct the complement of mask matrix
    for currpair in range(len(xy)):
        xc, yc = xy[currpair]
        zm += np.exp(-((x - xc)**2) / (2 * sx**2) -
                     ((y - yc)**2) / (2 * sy**2))

    fltrmat = np.abs(nft.ifft2((1 - zm) * ftmat))

    return fltrmat
