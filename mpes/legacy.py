#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from __future__ import print_function, division
import re
import glob as g
import numpy as np
from numpy import polyval as poly
from scipy.interpolate import interp1d


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
