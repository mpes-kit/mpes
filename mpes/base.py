#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from __future__ import print_function, division
from . import utils as u
#from . import dictdump as dictdump
from silx.io import dictdump
import deepdish.io as dio
import numpy as np
import glob as g
import natsort as nts
import cv2
from functools import partial
import scipy.io as sio
from scipy.interpolate import griddata
import pickle


class FileCollection(object):
    """ File collecting and sorting class.
    """

    def __init__(self, files=[], file_sorting=True, folder=''):

        self.sorting = file_sorting
        self.files = self._sort_terms(files, self.sorting)
        self.allfiles = self.files.copy()
        self.folder = folder

    def __add__(self, other):
        """ Combine two FileCollection instances (self, other) by combining the file names.
        """

        files = list(set(self.files) | set(other.files))

        return FileCollection(files=files, file_sorting=self.sorting)

    def __iter__(self):
        """ Iterator of the current class defined as iterating over file names.
        """

        for file in self.files:
            yield file

    @property
    def nfiles(self):
        """ Total number of files for a specific workflow.
        """

        return len(self.files)

    @property
    def nallfiles(self):
        """ Total number of files with the same attributes.
        # of allfiles >= # of files (equal when no selection criterion is applied).
        """

        return len(self.allfiles)

    @property
    def fileID(self):
        """ The sequence IDs of the files.
        """

        return list(range(self.nfiles))

    @staticmethod
    def _sort_terms(terms, parameter):
        """
        Sort terms according to parameter value.

        **Parameters**\n
        terms: list
            List of terms (e.g. strings).
        parameter: bool
            Decision parameter for sorting.

        **Return**\n
            Sorted or unsorted terms.
        """

        if parameter == True:
            return nts.natsorted(terms)
        else:
            return terms

    def gather(self, folder='', identifier=r'/*.h5', f_start=None, f_end=None, f_step=1, file_sorting=True):
        """
        Gather files from a folder (specified at instantiation).

        **Parameters**\n
        identifier: str | r'/*.h5'
            File identifier used for glob.glob().
        f_start, f_end, f_step: int, int, int | None, None, 1
            Starting, ending file id and the step. Used to construct a file selector.
        file_sorting: bool | True
            Option to sort the files by their names.
        """

        f_start, f_end, f_step = u.intify(f_start, f_end, f_step)

        try:
            if folder == '':
                self.files = g.glob(self.folder + identifier)
            else:
                self.files = g.glob(folder + identifier)

            if file_sorting == True:
                self.files = self._sort_terms(self.files, file_sorting)

            self.allfiles = self.files.copy()
            self.files = self.files[slice(f_start, f_end, f_step)]

        except:
            raise Exception('No legitimate folder address is specified for file retrieval!')

    def filter(self, wexpr=None, woexpr=None, str_start=None, str_end=None):
        """ Filter filenames by keywords.

        **Parameters**\n
        wexpr: str | None
            Expression in a name to leave in the filename list (w = with).
        woexpr: str | None
            Expression in a name to leave out of the filename list (wo = without).
        str_start, str_end: int, int | None, None
            Starting and ending indices of the filename string.

        **Return**\n
        filteredFiles: list
            List of filtered filenames.
        """

        if (wexpr is None) and (woexpr is None):
            filteredFiles = self.files
        elif wexpr:
            filteredFiles = [i for i in self.files if wexpr in i[str_start:str_end]]
        elif woexpr:
            filteredFiles = [i for i in self.files if woexpr not in i[str_start:str_end]]

        return filteredFiles

    def select(self, ids=[], update='', ret=False):
        """ Select gathered files by the filename ids.

        **Parameters**\n
        ids: list/1D array | []
            File IDs for selection.
        update: str | ''
            File address list update condition,
            'remove' = remove the selected files.
            'keep' = keep the selected files and remove the rest.
            others strings or no action = do nothing.
            '' = nothing happens (can be used in combination with `ret=True`
            to retrieve filtered file names).
        ret: str | 'selected'
            Return option,
            :'selected': Return of the selected files.
            :'rest': Return the rest of the files (not selected).
            :False: No action (return).
        """

        if self.files:
            selectedFiles = list(map(self.files.__getitem__, ids))
            selectedFiles = self._sort_terms(selectedFiles, self.sorting)

            if update == 'remove':
                difflist = list(set(self.files) - set(selectedFiles))
                self.files = self._sort_terms(difflist, self.sorting)
            elif update == 'keep':
                self.files = selectedFiles

            if ret == 'selected':
                return selectedFiles
            elif ret == 'rest':
                return self.files

        else:
            raise ValueError('No file address has been gathered!')


class MapParser(FileCollection):
    """ Parser of recorded parameters and turn into functional maps.
    """

    def __init__(self, files=[], file_sorting=True, folder=None, **kwds):

        super().__init__(files=files, file_sorting=file_sorting, folder=folder)

    @property
    def bfile(self, **kwds):
        """ File containing the binning parameters.
        """

        fstr_b = kwds.pop('namestr', 'binning.')
        return self.filter(wexpr=fstr_b)[0] # Take the first file with the searched string

    @property
    def kfile(self, **kwds):
        """ File containing the momentum correction and calibration information.
        """

        fstr_k = kwds.pop('namestr', 'momentum.')
        return self.filter(wexpr=fstr_k)[0] # Take the first file with the searched string

    @property
    def Efile(self, **kwds):
        """ File containing the energy calibration information.
        """

        fstr_E = kwds.pop('namestr', 'energy.')
        return self.filter(wexpr=fstr_E)[0] # Take the first file with the searched string

    @staticmethod
    def listfind(namelist, name, itemlist):
        """ Find item in the itemlist according to the name index in the namelist.
        """

        return itemlist[namelist.index(name)]

    def parse_bfile(self):
        """ Retrieve the binning parameters.
        """

        binDict = dio.load(self.bfile)
        binaxes = list(map(lambda x: x.decode('utf-8'), binDict['binaxes'].tolist()))
        binranges = binDict['binranges'].tolist()
        binsteps = binDict['binsteps'].tolist()

        # Retrieve the binning steps along X and Y axes
        self.xstep = self.listfind(binaxes, 'X', binsteps)
        self.ystep = self.listfind(binaxes, 'Y', binsteps)

        # Retrieve the binning ranges (br) along X and Y axes
        self.xbr_start, self.xbr_end = self.listfind(binaxes, 'X', binranges)
        self.ybr_start, self.ybr_end = self.listfind(binaxes, 'Y', binranges)

    def parse_kmap(self, key='coeffs'):
        """ Retrieve the parameters to construct the momentum conversion function.
        """

        self.parse_bfile()
        self.fr, self.fc = dio.load(self.kfile)['calibration'][key]
        self.xcent, self.ycent = dio.load(self.kfile)['pcent']

    def parse_Emap(self, key='coeffs'):
        """ Retrieve the parameters to construct the energy conversion function.
        """

        self.poly_a = dio.load(self.Efile)['calibration'][key]

    def parse_wmap(self, key='warping'):
        """ Retrieve the parameters to construct the distortion correction function
        """

        self.warping = dio.load(self.kfile)[key]

    @staticmethod
    def parse(parse_map, **mapkeys):
        """ Parse map parameters stored in files.

        **Parameter**\n
        parse_map: function
            Run parse_map function to populate the class namespace.

        **Return**\n
        flag: int (0 or 1)
            Returns 1 if successful, 0 if not.
        """

        try:
            parse_map(**mapkeys)
            return 1 # Retrieved mapping parameters successfully
        except:
            return 0 # Failed to retrieve parameters

    @staticmethod
    def mapConstruct(mapfunc, **kwds):
        """ Construct mapping function (partial function) by filling in certain
        known parameters.
        """

        return partial(mapfunc, **kwds)

    @property
    def kMap(self, parse_key='coeffs', **kwds):
        """ The (row, column) to momentum coordinate transform function.
        """

        # Load the parameters to construct the momentum conversion function
        ret = self.parse(self.parse_kmap, key=parse_key)
        if ret == 1:

            kmap = kwds.pop('map', detrc2krc)
            if kwds: # Parse other remaining keyword arguments
                return self.mapConstruct(kmap, **kwds)
            else:
                return self.mapConstruct(kmap, fr=self.fr, fc=self.fc,
                rstart=self.ybr_start, cstart=self.xbr_start, rstep=self.ystep, cstep=self.xstep)

        else:
            return None

    @property
    def EMap(self, parse_key='coeffs', **kwds):
        """ The ToF to energy coordinate transform function.

        **Parameter**\n
        parse_key: str | 'coeffs'
            Parsing key for extracting parameters.
        """

        # Load the parameters to construct the energy conversion function
        ret = self.parse(self.parse_Emap, key=parse_key)
        if ret == 1:

            Emap = kwds.pop('map', tof2evpoly)
            if kwds: # Parse other remaining keyword arguments
                return self.mapConstruct(Emap, **kwds)
            else:
                return self.mapConstruct(Emap, a=self.poly_a)

        else:
            return None

    @property
    def wMap(self, parse_key='warping', **kwds):
        """ The distortion correction transform function.
        """

        # Load warping transformation
        ret = self.parse(self.parse_wmap, key=parse_key)
        if ret == 1:

            wmap = kwds.pop('map', perspectiveTransform)
            if kwds: # Parse other remaining keyword arguments
                return self.mapConstruct(wmap, **kwds)
            else:
                return self.mapConstruct(wmap, M=self.warping)

        else:
            return None


def saveClassAttributes(clss, form, save_addr):
    """ Save class attributes.

    **Parameters**\n
    clss: instance
        Handle of the instance to be saved.
    form: str
        Format to save in ('h5'/'hdf5', 'mat', or 'dmp'/'dump').
    save_addr: str
        The address to save the attributes in.
    """

    save_addr = u.appendformat(save_addr, form)

    if form == 'mat':
        sio.savemat(save_addr, clss.__dict__)

    elif form in ('h5', 'hdf5'):
        try:
            dictdump.dicttoh5(clss.__dict__, save_addr)
        except:
            dio.save(save_addr, clss.__dict__, compression=None)
            
    elif form in ('dmp', 'dump'):
        fh = open(save_addr, 'wb')
        pickle.dump(clss, fh)
        fh.close()

    else:
        raise NotImplementedError
        
        
def loadClassAttributes(form, load_addr):
    """ Load class from saved attributes.

    **Parameters**\n
    form: dump
        Format to load from (only works for 'dump' right now).
    load_addr: str
        The address to load the attributes from.
    
    **Return**\n
        Returns the loaded class
    """

    load_addr = u.appendformat(load_addr, form)
        
    if form in ('dmp', 'dump'):
        fh = open(load_addr, 'rb')
        clss = pickle.load(fh)
        fh.close()

    else:
        raise NotImplementedError
        
    return clss


def tof2evpoly(a, E0, t):
    """
    Polynomial approximation of the time-of-flight to electron volt
    conversion formula.

    **Parameters**\n
    a: 1D array
        Polynomial coefficients.
    E0: float
        Energy offset.
    t: numeric array
        Drift time of electron.

    **Return**\n
    E: numeric array
        Converted energy
    """

    odr = len(a) # Polynomial order
    a = a[::-1]
    E = 0

    for i, d in enumerate(range(1, odr+1)):
        E += a[i]*t**d
    E += E0

    return E


def imxy2kxy(x, y, x0, y0, fx, fy):
    """
    Conversion from Cartesian coordinate in binned image (x, y) to momentum coordinates (kx, ky).

    **Parameters**\n
    x, y: numeric, numeric
        Components of the Cartesian coordinates.
    x0, y0: numeric, numeric
        Origins of the Cartesian coordinates.
    fx, fy: numeric numeric
        Numerical scaling in the conversion.
    """

    kx = fx * (x - x0)
    ky = fy * (y - y0)

    return (kx, ky)


def detxy2kxy(xdet, ydet, xstart, ystart, x0, y0, fx, fy, xstep, ystep):
    """
    Conversion from detector coordinates (xd, yd) to momentum coordinates (kx, ky).

    **Parameters**\n
    xdet, ydet: numeric, numeric
        Pixel coordinates in the detector coordinate system.
    xstart, ystart: numeric, numeric
        The starting pixel number in the detector coordinate system
        along the x and y axes used in the binning.
    x0, y0: numeric, numeric
        The center pixel position in binned image used in calibration.
    fx, fy: numeric, numeric
        Scaling factor along the x and y axes (in binned image).
    xstep, ystep: numeric, numeric
        Binning step size along x and y directions.
    """

    xdet0 = xstart + xstep * x0
    ydet0 = ystart + ystep * y0
    kx = fx * ((xdet - xdet0) / xstep)
    ky = fy * ((ydet - ydet0) / ystep)

    return (kx, ky)


def imrc2krc(r, c, r0, c0, fr, fc):
    """
    Conversion from image coordinate (row, column) to momentum coordinates (kr, kc).
    """

    kr = fr * (r - r0)
    kc = fc * (c - c0)

    return (kr, kc)


def detrc2krc(rdet, cdet, rstart, cstart, r0, c0, fr, fc, rstep, cstep):
    """
    Conversion from detector coordinates (rdet, cdet) to momentum coordinates (kr, kc).
    """

    rdet0 = rstart + rstep * r0
    cdet0 = cstart + cstep * c0
    kr = fr * ((rdet - rdet0) / rstep)
    kc = fc * ((cdet - cdet0) / cstep)

    return (kr, kc)


def dfieldapply(edf, dfield, X='X', Y='Y', newX='Xm', newY='Ym'):
    """
    Application of the inverse displacement-field to the dataframe coordinates
    """
    
    x = edf[X]
    y = edf[Y]
     
    edf[newX], edf[newY] = dfield[0,np.int16(x),np.int16(y)], dfield[1,np.int16(x),np.int16(y)]
    return edf

def generateDfield(rdeform_field, cdeform_field):
    """
    Generate inverse deformation field using inperpolation with griddata.
    Assuming the binning range of the input ``rdeform_field`` and ``cdeform_field`` covers the whole detector.

    **Parameters**\n
    rdeform_field, cdeform_field: 2d array, 2d array
        Row-wise and column-wise deformation fields.
    """
    # Interpolate to 2048x2048 grid of the detector coordinates
    grid_x, grid_y = np.mgrid[0:cdeform_field.shape[0]:(cdeform_field.shape[0]/2048), 0:cdeform_field.shape[1]:(cdeform_field.shape[1]/2048)]
    XY = []
    Z = []
    for i in np.arange(cdeform_field.shape[0]):
        for j in np.arange(cdeform_field.shape[1]):  
            XY.append([rdeform_field[i,j],cdeform_field[i,j]])
            Z.append(2048/cdeform_field.shape[0]*i)
            
    inv_rdeform_field = griddata(np.asarray(XY), Z, (grid_x, grid_y))

    XY = []
    Z = []
    for i in np.arange(cdeform_field.shape[0]):
        for j in np.arange(cdeform_field.shape[1]):  
            XY.append([rdeform_field[i,j],cdeform_field[i,j]])
            Z.append(2048/cdeform_field.shape[1]*j)
            
    inv_cdeform_field = griddata(np.asarray(XY), Z, (grid_x, grid_y))

    # TODO: what to do about the nans at the boundary? leave or fill with zeros?
    #inv_rdeform_field = np.nan_to_num(inv_rdeform_field)
    #inv_rdeform_field = np.nan_to_num(inv_cdeform_field)
    
    dfield = np.asarray([inv_rdeform_field, inv_cdeform_field])
    
    return dfield


def reshape2d(data, apply_axis):
    """
    Reshape matrix to apply 2D function to.

    **Parameters**\n
    data: numpy.ndarray
        N-dimensional numerical array.
    apply_axis: tuple/list of int
        The index of the axes to apply the transform to.

    **Return**\n
    data: numpy.ndarray
        Reshaped n-dimensional array.
    """

    nax = len(apply_axis)
    dshape = data.shape
    dim_rest = tuple(set(range(data.ndim)) - set(apply_axis))
    shapedict = dict(enumerate(dshape))

    # Calculate the matrix dimension to reshape original data into
    reshapedict = {}
    for ax in apply_axis:
        reshapedict[ax] = dshape[ax]

    squeezed_dim = 1
    for dr in dim_rest:
        squeezed_dim *= shapedict[dr]
    reshapedict[nax+1] = squeezed_dim

    reshape_dims = tuple(reshapedict.values())
    data = data.reshape(reshape_dims)

    return data


def mapping(data, f, **kwds):
    """ Mapping a generic function to multidimensional data with
    the possibility to supply keyword arguments.

    **Parameter**\n
    data: numpy.ndarray
        Data to map the function to.
    f: function
        Function to map to data.
    **kwds: keyword arguments
        Keyword arguments of the function map.
    """

    result = np.asarray(list(map(lambda x:f(x, **kwds), data)))

    return result


def correctnd(data, warping, func=cv2.warpPerspective, **kwds):
    """ Apply a 2D transform to 2D in n-dimensional data.
    """

    apply_axis = kwds.pop('apply_axis', (0, 1))
    dshape = data.shape
    dsize = kwds.pop('dsize', op.itemgetter(*apply_axis)(dshape))

    # Reshape data
    redata = reshape2d(data, apply_axis)
    redata = np.moveaxis(redata, 2, 0)
    redata = mapping(redata, func, M=warping, dsize=dsize, **kwds)
    redata = np.moveaxis(redata, 0, 2).reshape(dshape)

    return redata


def perspectiveTransform(x, y, M):
    """ Implementation of the perspective transform (homography) in 2D.

    **Parameters**\n
    x, y: numeric, numeric
        Pixel coordinates of the original point.
    M: 2d array
        Perspective transform matrix.

    **Return**\n
    xtrans, ytrans: numeric, numeric
        Pixel coordinates after projective/perspective transform.
    """

    denom = M[2, 0]*x + M[2, 1]*y + M[2, 2]
    xtrans = (M[0, 0]*x + M[0, 1]*y + M[0, 2]) / denom
    ytrans = (M[1, 0]*x + M[1, 1]*y + M[1, 2]) / denom

    return xtrans, ytrans
