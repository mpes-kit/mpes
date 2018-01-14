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
import numpy as np
import pandas as pd
import re, glob2
import numpy.fft as nft
from scipy.interpolate import interp1d
from numpy import polyval as poly
from scipy.signal import savgol_filter
import igor.igorpy as igor
from .igoribw  import loadibw
from PIL import Image as pim


# ================= #
# Utility functions #
# ================= #

def find_nearest(val, narray):
    """
    Find the value closest to a given one in a 1D array

    **Parameters**
    
    val : float
        Value of interest
    narray : 1D numeric array
        array to look for the nearest value

    **Return**

    ind : int
        index of the value nearest to the sepcified
    """

    return np.argmin(np.abs(narray - val))


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


def SortNamesBy(namelist, pattern):
    """
    Sort a list of names according to a particular sequence of numbers (specified by a regular expression search pattern)

    Parameters

    namelist : str
        List of name strings
    pattern : str
        Regular expression of the pattern

    Returns

    orderedseq : array
        Ordered sequence from sorting 
    sortednamelist : str
        Sorted list of name strings
    """

    # Extract a sequence of numbers from the names in the list
    seqnum = np.array([re.search(pattern, namelist[i]).group(1)
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


def binarysearch(arr, val):
    """
    Equivalent to BinarySearch(waveName, val) in Igor Pro, the sorting order is determined automatically
    """

    sortedarr = np.sort(arr)
    if np.array_equal(arr, sortedarr):
        return np.searchsorted(arr, val, side='left') - 1
    elif np.array_equal(arr, sortedarr[::-1]):
        return np.size(arr) - np.searchsorted(arr[::-1], val, side='left') - 1


def searchinterp(arr, val):
    """
    Equivalent to BinarySearchInterp(waveName, val) in Igor Pro, the sorting order is determined automatically
    """

    indstart = binarysearch(arr, val)
    indstop = indstart + 1
    indarray = np.array([indstart, indstop])
    finterp = interp1d(arr[indstart:indstop + 1], indarray, kind='linear')

    return finterp(val) + 0  # +0 because of data type conversion


def linterp(xind, yarr, frac):
    """
    Linear interpolation
    """

    return yarr[xind] * (1 - frac) + yarr[xind + 1] * frac


# ====================== #
#  File I/O and parsing  #
# ====================== #

def readtsv(fdir, header=None, dtype='float', **kwds):
    """
    Read tsv file from hemispherical detector
    
    ***Parameters***
    
    fdir : str
        file directory
    header : int | None
        number of header lines
    dtype : str | 'float'
        data type of the return numpy.ndarray
    **kwds : keyword arguments
        other keyword arguments for pandas.read_table()
        
    ***Return***
    
    data : numpy ndarray
        read and type-converted data
    """
    
    data = np.asarray(pd.read_table(fdir, delim_whitespace=True, \
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

    txtfiles = glob2.glob(ffolder + r'\*.txt')
    for ind, fname in enumerate(txtfiles):
        if keytext in fname:
            txtfile = txtfiles[ind]

    return txtfile


def parsenum(
        NumberPattern,
        strings,
        CollectorList,
        linenumber,
        offset=0,
        Range='all'):
    """
    Number parser for reading calibration file
    """

    # Specify Range as 'all' to take all numbers, specify number limits to
    # pick certain number
    numlist = re.findall(NumberPattern, strings[linenumber + offset])
    if Range == 'all':
        CollectorList.append(numlist)
    else:
        Rmin, Rmax = re.split(':', Range)
        # One-sided slicing with max value specified in number
        if Rmin == 'min':
            CollectorList.append(numlist[:int(Rmax)])
        # One-sided slicing with min value specified in number
        elif Rmax == 'max':
            CollectorList.append(numlist[int(Rmin):])
        # Two-sided slicing with bothe min and max specified in number
        else:
            CollectorList.append(numlist[int(Rmin):int(Rmax)])

    return CollectorList


def readLensModeParameters(calibfiledir, lensmode='WideAngleMode'):
    """
    Retrieve the calibrated lens correction parameters
    """

    # For wide angle mode
    if lensmode == 'WideAngleMode':

        LensModeDefaults, LensParamLines = [], []
        with open(calibfiledir, 'r') as fc:

            # Read the full file as a line-split string block
            calib = fc.read().splitlines()
            # Move read cursor back to the beginning
            fc.seek(0)
            # Scan through calibration file, find and append line indices
            # (lind) to specific lens settings
            for lind, line in enumerate(fc):
                if '[WideAngleMode defaults' in line:
                    LensModeDefaults.append(lind)
                elif '[WideAngleMode@' in line:
                    LensParamLines.append(lind)

        # Specify regular expression pattern for retrieving numbers
        numpattern = r'[-+]?\d*\.\d+|[-+]?\d+'

        # Read detector settings at specific lens mode
        aRange, eShift = [], []
        for linum in LensModeDefaults:

            # Collect the angular range
            aRange = parsenum(
                numpattern,
                calib,
                aRange,
                linenumber=linum,
                offset=2,
                Range='all')
            # Collect the eShift
            eShift = parsenum(
                numpattern,
                calib,
                eShift,
                linenumber=linum,
                offset=3,
                Range='all')

        # Read list calibrated Da coefficients at all retardation ratios
        rr, aInner, Da1, Da3, Da5, Da7 = [], [], [], [], [], []
        for linum in LensParamLines:

            # Collect the retardation ratio (rr)
            rr = parsenum(
                numpattern,
                calib,
                rr,
                linenumber=linum,
                offset=0,
                Range='all')
            # Collect the aInner coefficient
            aInner = parsenum(
                numpattern,
                calib,
                aInner,
                linenumber=linum,
                offset=1,
                Range='all')
            # Collect Da1 coefficients
            Da1 = parsenum(
                numpattern,
                calib,
                Da1,
                linenumber=linum,
                offset=2,
                Range='1:4')
            # Collect Da3 coefficients
            Da3 = parsenum(
                numpattern,
                calib,
                Da3,
                linenumber=linum,
                offset=3,
                Range='1:4')
            # Collect Da5 coefficients
            Da5 = parsenum(
                numpattern,
                calib,
                Da5,
                linenumber=linum,
                offset=4,
                Range='1:4')
            # Collect Da7 coefficients
            Da7 = parsenum(
                numpattern,
                calib,
                Da7,
                linenumber=linum,
                offset=5,
                Range='1:4')

        aRange, eShift, rr, aInner = list(map(lambda x: np.asarray(
            x, dtype='float').ravel(), [aRange, eShift, rr, aInner]))
        Da1, Da3, Da5, Da7 = list(
            map(lambda x: np.asarray(x, dtype='float'), [Da1, Da3, Da5, Da7]))

        return aRange, eShift, rr, aInner, Da1, Da3, Da5, Da7

    else:
        print('This mode is currently not supported!')


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
    Convert image to numpy ndarray
    """
    
    mat = np.asarray(pim.open(fdir))
    return mat


# =================== #
# Data transformation #
# =================== #

def MCP_Position_mm(Ek, Ang, aInner, Da):
    """
    In the region [-aInner, aInner], calculate the corrected isoline positions using
    the given formula in the SPECS HSA manual (p47 of SpecsLab, Juggler and CCDAcquire).
    In the region beyond aInner on both sides, use Taylor expansion and approximate
    the isoline position up to the first order, i.e.

    n = zInner + dAng*zInner'

    The np.sign() and abs() take care of the sign on each side and reduce the
    conditional branching to one line.
    """

    if abs(Ang) <= aInner:

        return zInner(Ek, Ang, Da)
    else:
        dA = abs(Ang) - aInner
        return np.sign(Ang) * (zInner(Ek, aInner, Da) +
                               dA * zInner_Diff(Ek, aInner, Da))


def zInner(Ek, Ang, Da):
    """
    Calculate the isoline position by interpolated polynomial at a certain kinetic energy
    (Ek) and photoemission angle (Ang).
    """
    D1, D3, D5, D7 = Da

    return poly(D1, Ek) * (Ang) + 10**(-2) * poly(D3, Ek) * (Ang)**3 + \
        10**(-4) * poly(D5, Ek) * (Ang)**5 + 10**(-6) * poly(D7, Ek) * (Ang)**7


def zInner_Diff(Ek, Ang, Da):
    """
    Calculate the derivative of the isoline position by interpolated polynomial at a
    certain kinetic energy (Ek) and photoemission angle (Ang).
    """

    D1, D3, D5, D7 = Da

    return poly(D1, Ek) + 3*10**(-2)*poly(D3, Ek)*(Ang)**2 + \
        5*10**(-4)*poly(D5, Ek)*(Ang)**4 + 7*10**(-6)*poly(D7,Ek)*(Ang)**6


def slice2d(datamat, xaxis, yaxis, xmin, xmax, ymin, ymax):
    """
    Slice ARPES matrix (E,k) according to specified energy and angular ranges
    """

    axes = [xaxis, xaxis, yaxis, yaxis]
    bounds = [xmin, xmax, ymin, ymax]
    lims = list(map(lambda x, y: find_nearest(x, y), bounds, axes))
    slicedmat = datamat[lims[0]:lims[1], lims[2]:lims[3]]

    return slicedmat, xaxis[lims[0]:lims[1]], yaxis[lims[2]:lims[3]]


def slice3d(datamat, xaxis, yaxis, zaxis, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Slice trARPES matrix (E,k,t) according to specified energy and angular ranges
    """

    axes = [xaxis, xaxis, yaxis, yaxis, zaxis, zaxis]
    bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
    lims = list(map(lambda x, y: find_nearest(x, y), bounds, axes))
    slicedmat = datamat[lims[0]:lims[1], lims[2]:lims[3], lims[4]:lims[5]]

    return slicedmat, xaxis[lims[0]:lims[1]
                            ], yaxis[lims[2]:lims[3]], zaxis[lims[4]:lims[5]]


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
