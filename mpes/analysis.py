#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""
# =======================================
# Sections:
# 1.  Background removal
# 2.  Coordinate calibration
# 3.  Image segmentation
# 4.  Image correction
# 5.  Fitting routines
# 6.  Fitting result parsing and testing
# =======================================

from __future__ import print_function, division
from . import base, utils as u, ellipsefit as elf
from math import cos, pi
import numpy as np
from numpy.linalg import norm, lstsq
import scipy.optimize as opt
from scipy.special import wofz
import scipy.io as sio
import pandas as pd
from skimage import measure, filters, morphology
from skimage.draw import line, circle, polygon
from skimage.feature import peak_local_max
import cv2
import astropy.stats as astat
import photutils as pho
from symmetrize import sym, pointops as po
from functools import reduce, partial
from funcy import project
import operator as op
import matplotlib.pyplot as plt
import bokeh.plotting as pbk
from bokeh.io import output_notebook
from bokeh.palettes import Category10 as ColorCycle
import itertools as it
from tqdm import tqdm
import warnings as wn


# ==================== #
#  Background removal  #
# ==================== #

def shirley(x, y, tol=1e-5, maxiter=20, explicit=False, warning=False):
    """
    Calculate the 1D best Shirley-Proctor-Sherwood background S for a dataset (x, y).
    A. Proctor, P. M. A. Sherwood, Anal. Chem. 54 13 (1982).
    The function is adapted from Kane O'Donnell's routine
    1. Finds the biggest peak
    2. Use the minimum value on either side of this peak as the terminal points
    of the Shirley background.
    3. Iterate over the process within maximum allowed iteration (maxiter) to
    reach the tolerance level (tol).

    **Parameters**

    x : 1D numeric array
        the photoelectron energy axis
    y : 1D numeric array
        the photoemission intensity axis
    tol : float | 1e-5
        fitting tolerance
    maxiter : int | 20
        maximal iteration
    explicit : bool | False
        explicit display of iteration number
    warning : bool | False
        display of warnings during calculation

    **Return**

    sbg : 1D numeric array
        Calculated Shirley background
    """

    # Set the energy values in decreasing order
    if x[0] < x[-1]:
        is_reversed = True
        x = x[::-1]
        y = y[::-1]
    else:
        is_reversed = False

    # Locate the biggest peak
    maxidx = abs(y - np.amax(y)).argmin()

    # If maxidx is either end of the spectrum, this algorithm cannot be
    # used, return a zero background instead
    if maxidx == 0 or maxidx >= len(y) - 1:
        if warning == True:
            print("Boundaries too high for algorithm: returning a zero background.")
        return np.zeros(x.shape)

    # Locate the minima either side of maxidx.
    lmidx = abs(y[0:maxidx] - np.amin(y[0:maxidx])).argmin()
    rmidx = abs(y[maxidx:] - np.amin(y[maxidx:])).argmin() + maxidx
    xl, yl = x[lmidx], y[lmidx]
    xr, yr = x[rmidx], y[rmidx]

    # Max integration index
    imax = rmidx - 1

    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below lmidx and initially zero above
    B = np.zeros(x.shape)
    B[:lmidx] = yl - yr
    Bnew = B.copy()

    niter = 0
    while niter < maxiter:
        if explicit:
            print("Iteration = " + str(it))

        # Calculate the new k factor (background strength)
        ksum = 0.0
        for i in range(lmidx, imax):
            ksum += (x[i] - x[i + 1]) * 0.5 * (y[i] + y[i + 1]
                                               - 2 * yr - B[i] - B[i + 1])
        k = (yl - yr) / ksum

        # Calculate the new B (background shape) at every x position
        for i in range(lmidx, rmidx):
            ysum = 0.0
            for j in range(i, imax):
                ysum += (x[j] - x[j + 1]) * 0.5 * (y[j] + y[j + 1]
                                                   - 2 * yr - B[j] - B[j + 1])
            Bnew[i] = k * ysum

        # Test convergence criterion
        if norm(Bnew - B) < tol:
            B = Bnew.copy()
            break
        else:
            B = Bnew.copy()
        niter += 1

    if niter >= maxiter and warning == True:
        print("Maximal iterations exceeded before convergence.")

    if is_reversed:
        return (yr + B)[::-1]
    else:
        return yr + B


def shirley2d(x, y, tol=1e-5, maxiter=20, explicit=False,
            warning=False):
    """
    2D Shirley background removal
    """

    nx = y.shape[0]
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')

    # Set the energy values in decreasing order
    if x[0] < x[-1]:
        is_reversed = True
        x = x[::-1]
        y = y[:, ::-1]
    else:
        is_reversed = False

    # Locate the biggest peak
    maxidx = abs(y - np.atleast_2d(np.amax(y, axis=1)).T).argmin(axis=1)
    maxex = maxidx.max()

    lmidx = abs(y[:, 0:maxex] - np.atleast_2d(np.amin(y[:, 0:maxex], axis=1)).T).argmin(axis=1)
    rmidx = abs(y[:, maxex:] - np.atleast_2d(np.amin(y[:, maxex:], axis=1)).T).argmin(axis=1) + maxex

    lmex, rmex = lmidx.min(), rmidx.max()

    xl, yl = x[lmidx], y[np.arange(nx, dtype='int64'), lmidx]
    xr, yr = x[rmidx], y[np.arange(nx, dtype='int64'), rmidx]

    # Max integration index
    imax = rmidx - 1
    mx = imax.max()

    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below lmidx and initially zero above
    B = np.zeros(y.shape, dtype='float64')
    for i in range(nx):
        B[i, :lmidx[i]] = yl[i] - yr[i]
    Bnew = B.copy()

    niter = 0
    while niter < maxiter:

        if explicit:
            print("Iteration = " + str(it))

        # Calculate the new k factor (background strength)
        ksum = np.zeros_like(yl)
        #int(lmidx.mean())
        for i in range(lmex, mx):
            ksum += (x[i] - x[i + 1]) * 0.5 * (y[:, i] + y[:, i + 1]\
                                               - 2 * yr - B[:, i] - B[:, i + 1])
        k = (yl - yr) / ksum

        # Calculate the new B (background shape) at every x position
        for i in range(lmex, rmex):
            ysum = np.zeros_like(yl)
            for j in range(i, mx):
                ysum += (x[j] - x[j + 1]) * 0.5 * (y[:, j] + y[:, j + 1]\
                                                   - 2 * yr - B[:, j] - B[:, j + 1])
            Bnew[:, i] = k * ysum

        dev = norm(Bnew - B)

        # Update B values
        B = Bnew.copy()

        # Test convergence criterion
        if dev < tol:
            break
        niter += 1

    if niter >= maxiter and warning == True:
        print("Maximal iterations exceeded before convergence.")

    if is_reversed:
        return (yr[:,np.newaxis] + B)[::-1]
    else:
        return yr[:,np.newaxis] + B


# 1D peak detection algorithm adapted from Sixten Bergman
# https://gist.github.com/sixtenbe/1178136#file-peakdetect-py
def _datacheck_peakdetect(x_axis, y_axis):
    """
    Input format checking
    """

    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise ValueError(
                "Input vectors y_axis and x_axis must have same length")

    # Needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)

    return x_axis, y_axis


def peakdetect1d(y_axis, x_axis = None, lookahead = 200, delta=0):
    """
    Function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively

    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    **Parameters**
    y_axis : list
        A list containing the signal over which to find peaks
    x_axis : list | None
        A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
    lookahead : int | 200
        distance to look ahead from a peak candidate to determine if
        it is the actual peak
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value
    delta : numeric | 0
        this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.

    **Returns**
    max_peaks : list
        positions of the positive peaks
    min_peaks : list
        positions of the negative peaks
    """

    max_peaks = []
    min_peaks = []
    dump = [] # Used to pop the first hit which almost always is false

    # Check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # Store data length for later use
    length = len(y_axis)


    # Perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")

    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x

        if y < mn:
            mn = y
            mnpos = x

        # Find local maxima
        if y < mx-delta and mx != np.Inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:

                max_peaks.append([mxpos, mx])
                dump.append(True)
                # Set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf

                if index+lookahead >= length:
                    # The end is within lookahead no more peaks can be found
                    break
                continue
            #else:
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        # Find local minima
        if y > mn+delta and mn != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:

                min_peaks.append([mnpos, mn])
                dump.append(False)
                # Set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf

                if index+lookahead >= length:
                    # The end is within lookahead no more peaks can be found
                    break
            #else:
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]


    # Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump

    except IndexError: # When no peaks have been found
        pass

    max_peaks = np.asarray(max_peaks)
    min_peaks = np.asarray(min_peaks)

    return max_peaks, min_peaks


def peakdetect2d(img, method='daofind', **kwds):
    """
    Peak-like feature detection in a 2D image.

    :Parameters:
        img : 2D array
            Image matrix.
        method : str | 'daofind'
            Detection method ('daofind' or 'maxlist').
        **kwds : keyword arguments
            Arguments passed to the specific methods chosen.

            :daofind: See `astropy.stats.sigma_clipped_stats()` and `photutils.detection.DAOStarFinder()`.
            sigma : float | 5.0
                Standard deviation of the clipping Gaussian.
            fwhm : float | 3.0
                FWHM of the convoluting Gaussian kernel.
            threshfactor : float | 8
                Intensity threshold for background-foreground separation (foreground is above threshold).

            :maxlist: See `skimage.feature.peak_local_max()`.
            mindist : float | 10
                Minimal distance between two local maxima.
            numpeaks : int | 7
                Maximum number of detected peaks.

    :Return:
        pks : 2D array
            Pixel coordinates of detected peaks, in (column, row) ordering.
    """

    if method == 'daofind':

        sg = kwds.pop('sigma', 5.0)
        fwhm = kwds.pop('fwhm', 3.0)
        threshfactor = kwds.pop('threshfactor', 8)

        mean, median, std = astat.sigma_clipped_stats(img, sigma=sg)
        daofind = pho.DAOStarFinder(fwhm=fwhm, threshold=threshfactor*std)
        sources = daofind(img)
        pks = np.stack((sources['ycentroid'], sources['xcentroid']), axis=1)

    elif method == 'maxlist':

        mindist = kwds.pop('mindist', 10)
        numpeaks = kwds.pop('numpeaks', 7)

        pks = peak_local_max(img, min_distance=mindist, num_peaks=numpeaks)

    return pks


# ======================== #
#  Coordinate calibration  #
# ======================== #

def calibrateK(img, pxla, pxlb, k_ab, coordb=[0., 0.], ret='axes'):
    """
    Momentum axes calibration using the pixel positions of two symmetry points (a and b)
    and the absolute coordinate of a single point (b). All coordinates should be specified
    in the (row_index, column_index) format.

    :Parameters:
        img : 2D array
            An energy cut of the band structure.
        pxla, pxlb : list/tuple/1D array
            Pixel coordinates of the two symmetry points (a and b).
        k_ab : float
            The known momentum space distance between the two symmetry points.
        coordb : list/tuple/1D array
            Actual coordinate of the symmetry point b.
        ret : str | 'axes'
            Return type specification, options include 'axes', 'extent', 'coeffs', 'grid', 'func', 'all'.

    :Returns:
        k_row, k_col : 1D array
            Momentum coordinates of the row and column.
        axis_extent : list
            Extent of the two momentum axis (can be used directly in imshow).
        k_rowgrid, k_colgrid : 2D array
            Row and column mesh grid generated from the coordinates
            (can be used directly in pcolormesh).
    """

    nr, nc = img.shape
    pxla, pxlb = map(np.array, [pxla, pxlb])
    d_ab = norm(pxla - pxlb)
    ratio = k_ab / d_ab # Distance conversion factor

    # Calculate the row-wise conversion factor
    rowdist = range(nr) - pxlb[0]
    k_row = rowdist * ratio + coordb[0]

    # Calculate the column-wise conversion factor
    coldist = range(nc) - pxlb[1]
    k_col = coldist * ratio + coordb[1]

    # Calculate other return parameters
    pfunc = partial(base.imrc2krc, fr=ratio, fc=ratio)
    k_rowgrid, k_colgrid = np.meshgrid(k_row, k_col)

    # Assemble into return dictionary
    kcalibdict = {}
    kcalibdict['axis'] = (k_row, k_col)
    kcalibdict['extent'] = (k_col[0], k_col[-1], k_row[0], k_row[-1])
    kcalibdict['coeffs'] = (ratio, ratio)
    kcalibdict['grid'] = (k_rowgrid, k_colgrid)

    if ret == 'all':
        return kcalibdict
    elif ret == 'func':
        return pfunc
    else:
        return project(kcalibdict, ret)


def peaksearch(traces, tof, ranges=None, method='range-limited', pkwindow=3, plot=False):
    """
    Detect a list of peaks in the corresponding regions of multiple EDCs

    :Parameters:
        traces : 2D array
            Collection of EDCs.
        tof : 1D array
            Time-of-flight values.
        ranges : list of tuples | None
            List of ranges for peak detection.
        method : str | 'range-limited'
            Method for peak-finding ('range-limited' and 'alignment').
        pkwindow : int | 3
            Window width of a peak(amounts to lookahead in mpes.analysis.peakdetect1d).
        plot : bool | False
            Specify whether to display a custom plot of the peak search results.

    :Returns:
        pkmaxs : 1D array
            Collection of peak positions
    """

    pkmaxs = []

    if plot:
        plt.figure(figsize=(10, 4))

    if method == 'range-limited': # Peak detection within a specified range
        for rg, trace in zip(ranges, traces.tolist()):

            cond = (tof >= rg[0]) & (tof <= rg[1])
            trace = np.array(trace).ravel()
            tofseg, trseg = tof[cond], trace[cond]
            maxs, _ = peakdetect1d(trseg, tofseg, lookahead=pkwindow)
            pkmaxs.append(maxs[0, :])

            if plot:
                plt.plot(tof, trace, '--k', linewidth=1)
                plt.plot(tofseg, trseg, linewidth=2)
                plt.scatter(maxs[0, 0], maxs[0, 1], s=30)

    elif method == 'alignment':
        raise NotImplementedError

    pkmaxs = np.asarray(pkmaxs)

    return pkmaxs


def calibrateE(pos, vals, order=3, refid=0, ret='func', E0=None, t=None):
    """
    Energy calibration by nonlinear least squares fitting of spectral landmarks on
    a set of (energy dispersion curves (EDCs). This amounts to solving for the
    coefficient vector, a, in the system of equations T.a = b. Here T is the
    differential drift time matrix and b the differential bias vector, and
    assuming that the energy-drift-time relationship can be written in the form,
    E = sum (a_n * t**n) + E0

    :Parameters:
        pos : list/array
            Positions of the spectral landmarks (e.g. peaks) in the EDCs.
        vals : list/array
            Bias voltage value associated with each EDC.
        order : int | 3
            Polynomial order of the fitting function.
        refid : int | 0
            Reference data point index, varies from 0 to vals.size - 1.
        ret : str | 'func'
            Return type, including 'func', 'coeffs', 'full', and 'axis' (see below).
        E0 : float | None
            Constant energy offset.
        t : numeric array | None
            Drift time.

    :Returns:
        pfunc : partial function
            Calibrating function with determined polynomial coefficients (except the constant offset).
        coeffs : 1D array
            Fitted polynomial coefficients.
        sol : tuple
            Full solution output of the least squares (see numpy.linalg.lstsq).
        eVscale : numpy array
            Calibrated energy scale in eV.
    """

    vals = np.array(vals)
    nvals = vals.size

    if refid >= nvals:
        wn.warn('Reference index (refid) cannot be larger than the number of traces!\
                Reset to the largest allowed number.')
        refid = nvals - 1

    # Top-to-bottom ordering of terms in the T matrix
    termorder = np.delete(range(0, nvals, 1), refid)
    # Left-to-right ordering of polynomials in the T matrix
    polyorder = np.linspace(order, 1, order, dtype='int')

    # Construct the T (differential drift time) matrix, Tmat = Tmain - Tsec
    Tmain = np.array([pos[refid]**p for p in polyorder])
    # Duplicate to the same order as the polynomials
    Tmain = np.tile(Tmain, (nvals-1, 1))

    Tsec = []

    for to in termorder:
        Tsec.append([pos[to]**p for p in polyorder])
    Tsec = np.asarray(Tsec)
    Tmat = Tmain - Tsec

    # Construct the b vector (differential bias)
    bvec = vals[refid] - np.delete(vals, refid)

    # Solve for the a vector (polynomial coefficients) using least squares
    sol = lstsq(Tmat, bvec, rcond=None)
    a = sol[0]

    # Construct the calibrating function
    pfunc = partial(base.tof2evpoly, a)

    # Return results according to specification
    ecalibdict = {}
    if (E0 is not None) and (t is not None):
        ecalibdict['axis'] = pfunc(E0, t)
    ecalibdict['coeffs'] = a

    if ret == 'all':
        return ecalibdict
    elif ret == 'func':
        return pfunc
    else:
        return project(ecalibdict, ret)


class EnergyCalibrator(base.FileCollection):
    """
    Electron binding energy calibration workflow.
    """

    def __init__(self, biases=None, files=[], folder=None, file_sorting=True, traces=None, tof=None):
        """ Initialization of the EnergyCalibrator class can follow different ways,

        1. Initialize with all the file paths in a list
        1a. Use an hdf5 file containing all binned traces and tof
        1b. Use a mat file containing all binned traces and tof
        1c. Use the raw data hdf5 files
        2. Initialize with the folder path containing any of the above files
        3. Initialize with the binned traces and the time-of-flight
        """

        self.biases = biases
        self.tof = tof

        super().__init__(folder=folder, file_sorting=file_sorting, files=files)

        if traces is not None:
            self.traces = traces
        else:
            self.traces = 0

    @property
    def nfiles(self):
        """ The number of loaded files.
        """

        return len(self.files)

    def read(self, form='h5', tracename='', tofname='ToF'):
        """ Read traces (e.g. energy dispersion curves) from files.

        :Parameters:
            form : str | 'h5'
                Format of the files ('h5' or 'mat').
            tracename : str | ''
                Name of the group/attribute corresponding to the trace.
            tofname : str | 'ToF'
                Name of the group/attribute corresponding to the time-of-flight.
        """

        if form == 'h5':

            traces = []
            for f in self.files:
                traces = np.asarray(File(f).get(tracename))

            tof = np.asarray(File(f).get(tofname))

        elif form == 'mat':

            for f in self.files:
                traces = sio.loadmat(f)[tracename]
            self.traces = np.array(traces)

            self.tof = sio.loadmat(f)[tofname].ravel()

    def normalize(self, **kwds):
        """ Normalize the spectra along an axis.

        :Parameters:
            **kwds : keyword arguments
                See the keywords for `mpes.utils.normspec()`
        """

        self.traces_normed = u.normspec(*self.traces, **kwds)

    @staticmethod
    def findCorrespondence(self, sig_still, sig_mov, position, order=3):
        """ Determine the parametric map between two traces by alignment.
        """

        from ptw import ptw
        w, siglim, a = ptw.timeWarp(sig_still, sig_mov)

        return a

    def featureSelect(self, ranges, refid=None, traces=None, infer_others=True, **kwds):
        """ Select the equivalent landmarks among all traces.

        :Parameters:
            range :
            refid : int | None
            traces : 2d array | None
            infer_others : bool | True
        """

        self.ranges = ranges
        if traces is None:
            traces = self.traces

        if infer_others == True:
            ranges_inferred = 0
            self.ranges = ranges_inferred
            pass

        # Run peak detection for each trace within the specified range
        self.peaks = peaksearch(traces, self.tof, self.ranges, **kwds)

    def calibrate(self, refid=0, ret=['coeffs'], **kwds):
        """ Calibrate the energy scales using optimization methods.

        :Parameters:
            refid : int | 0
                The trace ID (an integer)
            ret : list | ['coeffs']
                Options for return values.
            **kwds : keyword arguments
                See possible keywords for `mpes.analysis.calibrateE()`
        """

        landmarks = kwds.pop('landmarks', self.peaks)[:, 0]
        biases = kwds.pop('biases', self.biases)
        calibret = kwds.pop('calib_ret', False)
        self.calibration = calibrateE(landmarks, biases, refid=refid, ret=ret, **kwds)

        if calibret == True:
            return self.calibration

    def view(self, traces, segs=None, ranges=None, peaks=None, ret=False, backend='matplotlib',
            linekwds={}, scatterkwds={}, legkwds={}, **kwds):
        """ Display a plot showing all traces with annotation.

        :Parameters:
            traces : 2d array
                Matrix of traces to visualize.
            segs :
            ranges :
            peaks : 2d array
                Peak positions for labelling the traces.
            ret : bool
                Return specification.
            backend : str | 'matplotlib'
                Backend specification, choose between 'matplotlib' (static) or 'bokeh' (interactive).
            linekwds : dict | {}
                Keyword arguments for line plotting (see `matplotlib.pyplot.plot()`).
            scatterkwds : dict | {}
                Keyword arguments for scatter plot (see `matplotlib.pyplot.scatter()`).
            legkwds : dict | {}
                Keyword arguments for legend (see `matplotlib.pyplot.legend()`).
            **kwds : keyword arguments
            ===============  ==========  ================================
            keyword          data type   meaning
            ===============  ==========  ================================
            maincolor        str
            labels           list        Labels for each curve
            xaxis            1d array    x (horizontal) axis values
            title            str         Title of the plot
            legend_location  str         Location of the plot legend
            ===============  ==========  ================================
        """

        maincolor = kwds.pop('maincolor', 'None')
        lbs = kwds.pop('labels', [str(b)+' V' for b in self.biases])
        xaxis = kwds.pop('xaxis', self.tof)
        ttl = kwds.pop('title', None)

        if backend == 'matplotlib':

            figsize = kwds.pop('figsize', (12, 4))
            f, ax = plt.subplots(figsize=figsize)
            for itr, trace in enumerate(traces):
                ax.plot(xaxis, trace, ls='--', linewidth=1, label=lbs[itr])

                # Emphasize selected EDC segments
                if (segs is not None) and (ranges is not None):
                    rg = ranges[itr]
                    cond = (self.tof >= rg[0]) & (self.tof <= rg[1])
                    tofseg, traceseg = self.tof[cond], trace[cond]
                    ax.plot(tofseg, traceseg, color='k', linewidth=2, **linekwds)
                # Emphasize extracted local maxima
                if peaks is not None:
                    ax.scatter(peaks[itr, 0], peaks[itr, 1], s=30, **scatterkwds)

            try:
                ax.legend(fontsize=12, **legkwds)
            except:
                pass

            ax.set_title(ttl)

        elif backend == 'bokeh':

            output_notebook(hide_banner=True)
            colors = it.cycle(ColorCycle[10])
            ttp = [('(x, y)', '($x, $y)')]

            figsize = kwds.pop('figsize', (800, 300))
            f = pbk.figure(title=ttl, plot_width=figsize[0], plot_height=figsize[1], tooltips=ttp)
            # Plotting the main traces
            for itr, c in zip(range(len(traces)), colors):
                f.line(xaxis, traces[itr,:], color=c, line_dash='solid', line_width=1,
                        line_alpha=1, legend=lbs[itr], **kwds)

                # Emphasize selected EDC segments
                if (segs is not None) and (ranges is not None):
                    rg = ranges[itr]
                    cond = (self.tof >= rg[0]) & (self.tof <= rg[1])
                    tofseg, traceseg = self.tof[cond], trace[cond]
                    f.line(tofseg, traceseg, color='k', line_width=2, **linekwds)

                # Plot detected peaks
                if peaks is not None:
                    f.scatter(peaks[itr, 0], peaks[itr, 1], fill_color=c, fill_alpha=0.8,
                                line_color=None, **scatterkwds)

            f.legend.location = kwds.pop('legend_location', 'top_right')
            f.legend.spacing= 0
            f.legend.padding = 2

            pbk.show(f)

        # ax.set_xlabel('Energy (eV)', fontsize=15)

        if ret:
            try:
                return f, ax
            except:
                return f

    def saveParameters(self, form='h5', save_addr='./energy'):
        """
        Save all the attributes of the workflow instance for later use
        (e.g. energy scale conversion).

        :Parameters:
            form : str | 'h5'
                The file format to save the attributes in ('h5'/'hdf5' or 'mat').
            save_addr : str | './energy'
                The filename to save the files with.
        """

        base.saveClassAttributes(self, form, save_addr)


# ==================== #
#  Image segmentation  #
# ==================== #

def blocknorm(data, mavg_axis=0, blockwidth=1):
    """
    Block-thresholding 2D data

    ***Parameters***

    data : ndarray
        data to normalize
    mavg_axis : int | 0
        axis to move the block along
    blockwidth : int | 1
        width of the moving block

    ***Returns***

    datanorm : ndarray
        block-normalized data
    """

    datar = np.rollaxis(data, mavg_axis)
    nr, nc = datar.shape
    datanorm = np.zeros_like(datar)
    for bst in range(nr):
        bnd = bst + blockwidth
        datanorm[bst:bnd] = datar[bst:bnd,:]/np.max(datar[bst:bnd,:])

    return np.rollaxis(datanorm, mavg_axis)


def segment2d(img, nbands=1, **kwds):
    """
    Electronic band segmentation using local thresholding
    and connected component labeling

    **Parameters**

    img : 2D numeric array
        the 2D matrix to segment
    nbands : int
        number of electronic bands
    **kwds : keyword arguments

    **Return**

    imglabeled : 2D numeric array
        labeled mask
    """

    ofs = kwds.pop('offset', 0)

    nlabel  =  0
    dmax = u.to_odd(max(img.shape))
    i = 0
    blocksize = dmax - 2 * i

    while (nlabel != nbands) or (blocksize <= 0):

        binadpt = filters.threshold_local(
    img, blocksize, method='gaussian', offset=ofs, mode='reflect')
        imglabeled, nlabel = measure.label(img > binadpt, return_num=True)
        i += 1
        blocksize = dmax - 2 * i

    return imglabeled


def ridgeDetect(mask, method='mask_mean_y', **kwds):
    """
    Detect the band ridges using selected methods.

    **Parameters**

    mask : numeric 2D array
        the 2D integer-valued mask with labeled bands
    method : str
        the method used for ridge detection
        'mask_mean_y' : mean mask position along y direction (default)
        'mask_mean_x' : mean mask position along x direction
    **kwds : keyword arguments
        ======= ========= ===================
        keyword data type meaning
        ======= ========= ===================
        x       int/float x axis coordinates
        y       int/float y axis coordinates
        ======= ========= ===================

    **Return**

    ridges : list of dataframes
        the ridge coordinates
    """

    # Collect input arguments
    nr, nc = mask.shape
    xaxis = kwds.pop('x', range(nr))
    yaxis = kwds.pop('y', range(nc))
    labels = np.unique(mask)
    nzlabels = labels[labels > 0]

    # Construct coordinate matrices
    xcoord, ycoord = np.meshgrid(xaxis, yaxis)
    xcoord, ycoord = xcoord.ravel(), ycoord.ravel()

    # Select the masked band region
    bands_df = pd.DataFrame(np.vstack((xcoord, ycoord, mask[xcoord, ycoord].ravel())).T, columns=['x','y','val'])
    bands_df = bands_df[bands_df['val'] > 0].reset_index(drop=True)

    # Calculate the coordinates of ridges for each electronic band
    ridges = []
    for lb in nzlabels:

        if method == 'mask_mean_y':
            band = bands_df[bands_df.val == lb].groupby(['val','x']).agg(['mean','min']).reset_index()
            # Terminate the band at certain condition
            band = band[band['y']['min'] > 0]
        elif method == 'mask_mean_x':
            band = bands_df[bands_df.val == lb].groupby(['val','y']).agg(['mean','min']).reset_index()
            # Terminate the band at certain condition
            band = band[band['x']['min'] > 0]

        ridges.append(band)

    return ridges


def regionExpand(mask, **kwds):
    """
    Expand the region of a binarized image around a line position

    **Parameters**

    mask : numeric binarized 2D array
        the mask to be expanded
    **kwds : keyword arguments
        =============  ==========  ===================================
        keyword        data type   meaning
        =============  ==========  ===================================
        method         str         method of choice ('offset', 'growth')
        value          numeric     value to be assigned to the masked
        linecoords     2D array    contains x and y positions of the line
        axoffsets      tuple/list  [downshift upshift] pixel number
        clipbounds     tuple/list  bounds in the clipping direction
        selem          ndarray     structuring element
        =============  ==========  ===================================
    **Return**

    mask : numeric 2D array
        modified mask (returns the original mask if insufficient arguments
        are provided for the chosen method for region expansion)
    """

    method = kwds.pop('method', 'offset')
    val = kwds.pop('value', 1)

    if method == 'offset':

        try:
            xpos, ypos = kwds.pop('linecoords')
            downshift, upshift = kwds.pop('axoffsets')
            lbl, lbu, ubl, ubu = kwds.pop('clipbounds', [0, np.inf, 0, np.inf])
            lb, ub = np.clip(ypos - downshift, lbl, lbu).astype('int'), np.clip(ypos + upshift, ubl, ubu).astype('int')
            for ind, x in enumerate(xpos):
                mask[x, lb[ind]:ub[ind]] = val
        except KeyError:
            print('Please specify the line coordinates and axis offsets!')

    elif method == 'growth':

        try:
            selem = kwds.pop('selem')
            mask = val*morphology.binary_dilation(mask, selem=selem)
        except KeyError:
            print('Please specify a structuring element for dilation!')

    return mask


def _signedmask(imr, imc, maskr, maskc, sign):
    """ Generate a binary mask using the masked coordinates

    :Parameters:
        imr, imc : int
            Row and column size of the image
        maskr, maskc : 1D array
            Row and column coordinates of the masked pixels
        sign : int
            Binary sign of the masked region, (0, 1)

    :Return:
        mask : 2D array
            Mask matrix
    """

    if sign == 1:
        mask = np.zeros((imr, imc))
        mask[maskr, maskc] = 1

    elif sign == 0:
        mask = np.ones((imr, imc))
        mask[maskr, maskc] = 0

    return mask


def circmask(img, rcent, ccent, rad, sign=1, ret='mask', **kwds):
    """ Use a circular binary mask to cover an image
    :Parameters:
        img : 2D array
            Input image to be masked
        rcent : float
            Row center position
        ccent : float
            Column center position
        rad : float
            Radius of circle
        sign : int | 1
            Binary sign of the masked region
        ret : str | 'mask'
            Return type ('mask', 'masked_image')
        kwds : keyword arguments
        =============  ==========  ============ =========================
           keyword     data type   default      meaning
        =============  ==========  ============ =========================
            shape      tuple/list  shape of img see skimage.draw.circle()
           method         str       'graphic'   'graphic' or 'algebraic'
          edgefactor     float        1.02       prefactor to rad**2
        =============  ==========  ============ =========================

    :Return:
        cmask or cmask*img : 2D array
            Mask only or masked image
    """

    rim, cim = img.shape
    shape = kwds.pop('shape', (rim, cim))
    method = kwds.pop('method', 'graphic')
    edgefac = kwds.pop('edgefactor', 1.02)

    # Generate circular mask of the chosen sign
    if method == 'graphic':
        rr, cc = circle(rcent, ccent, rad, shape=shape)
    elif method == 'algebraic':
        cmesh, rmesh = np.meshgrid(range(cim), range(rim))
        rr, cc = np.where((cmesh - ccent)**2 + (rmesh - rcent)**2 <= edgefac*rad**2)

    cmask = _signedmask(rim, cim, rr, cc, sign=sign)

    if ret == 'mask':
        return cmask
    elif ret == 'masked_image':
        return cmask*img
    elif ret == 'all':
        return cmask, cmask*img, [rr, cc]


def rectmask(img, rcent, ccent, shift, direction='row', sign=1, ret='mask', **kwds):
    """ Use a rectangular binary mask to cover an image

    :Parameters:
        img : 2D array
            Input image to be masked
        rcent : float
            Row center position
        ccent : float
            Column center position
        shift : int/list of int
            Pixel shifts
        direction : str | 'row'
            Direction to apply the shift to, 'row' or 'column' indicates row-wise
            or column-wise shift for generating the rectangular mask
        sign : int | 1
            Binary sign of the masked region
        ret : str | 'mask'
            Return type ('mask', 'masked_image')
        **kwds : keyword arguments

    :Return:
        cmask or cmask*img : 2D array
            Mask only or masked image
    """

    rim, cim = img.shape
    shape = kwds.pop('shape', (rim, cim))

    shift = np.asarray([shift]).ravel()
    if len(shift) == 1:
        neg_shift, pos_shift = shift, shift
    elif len(shift) == 2:
        neg_shift, pos_shift = shift

    # Calculate the vertices of the triangle
    if direction == 'row':

        # Along the row direction
        rverts = [rcent-neg_shift, rcent+pos_shift, rcent+pos_shift, rcent-neg_shift]
        cverts = [0, 0, cim, cim]

    elif direction == 'column':

        # Along the column direction
        rverts = [0, 0, rim, rim]
        cverts = [ccent-neg_shift, ccent+pos_shift, ccent+pos_shift, ccent-neg_shift]

    rr, cc = polygon(rverts, cverts, shape=shape)
    rmask = _signedmask(rim, cim, rr, cc, sign=sign)

    if ret == 'mask':
        return rmask
    elif ret == 'masked_image':
        return rmask*img
    elif ret == 'all':
        return rmask, rmask*img, [rr, cc]


def apply_mask_along(arr, mask, axes=None):
    """
    Apply a mask in a low dimensional slice throughout a high-dimensional array

    :Parameters:
        arr : nD array
            Multidimensional array for masking.
        mask : nD array
            Mask to apply.
        axes : list/tuple of int | None
            The axes to apply the mask to.

    :Return:
        maskedarr : nD array
            Masked multidimensional array.
    """

    ndimmask = mask.ndim
    ndimarr = arr.ndim
    maskshape = list(mask.shape)
    maskedarr = arr.copy()

    # Mask with the same dimension, just multiply
    if ndimarr == ndimmask:
        maskedarr *= mask

    # Mask with lower dimension than matrix, broadcast first, then multiply
    elif (ndimarr > ndimmask) and axes:
        ndimaug = ndimarr - ndimmask # The number of dimensions that needs to be augmented
        maskedarr = np.moveaxis(maskedarr, axes, list(range(ndimaug)))
        maskaugdim = [1]*ndimaug + maskshape
        maskaug = mask.reshape(maskaugdim)
        maskedarr *= maskaug
        maskedarr = np.moveaxis(maskedarr, list(range(ndimaug)), axes)

    return maskedarr


def points2path(pointsr, pointsc, ret='separated'):
    """
    Calculate ordered pixel cooridnates along a path defined by specific intermediate points.
    The approach constructs the path using a set of line segments bridging the specified points,
    therefore it is also able to trace the sequence indices of these special points.

    :Parameters:
        pointsr, pointsc : list/tuple/array
            The row and column pixel coordinates of the special points along the sampling path.
        ret : str | 'separated'
            Specify if return combined ('combined') or separated ('separated') row and column coordinates.

    :Returns:
        polyr, polyc : 1D array
            Pixel coordinates along the path traced out sequentially.
        pid : 1D array
            Pointwise indices of the special lpoints.
    """

    pointsr = np.round(pointsr).astype('int')
    pointsc = np.round(pointsc).astype('int')
    npts = len(pointsr)

    polyr, polyc, pid = [], [], np.zeros((npts,), dtype='int')

    for i in range(npts-1):

        lsegr, lsegc = line(pointsr[i], pointsc[i], pointsr[i+1], pointsc[i+1])

        # Attached all but the last element to the coordinate list to avoid
        # double inclusion (from the beginning of the next line segment)
        polyr.append(lsegr[:-1])
        polyc.append(lsegc[:-1])
        pid[i+1] = len(lsegr[:-1]) + pid.max()

    # Concatenate all line segments comprising the path
    polyr, polyc = map(np.concatenate, (polyr, polyc))

    if ret == 'combined':
        return np.stack((polyr, polyc), axis=1), pid
    elif ret == 'separated':
        return polyr, polyc, pid


def bandpath_map(bsvol, pathr=None, pathc=None, path_coords=None, eaxis=2):
    """
    Extract band diagram map from volumetric data.

    :Parameters:
        bsvol : 3D array
            Volumetric band structure data.
        pathr, pathc : 1D array | None, None
            Row and column pixel coordinates of the band path (ignored if path_coords is given).
        path_coords : 2D array | None
            Combined row and column pixel coordinates of the band path.
        eaxis : int | 2
            Energy axis index.

    :Return:
        bpm : 2D array
            Band path map sampled from the volumetric data.
    """

    bsvol = np.moveaxis(bsvol, eaxis, 2)

    if path_coords is not None:
        axid = np.where(np.array(path_coords.shape) == 2)[0][0]
        pathr, pathc = np.split(path_coords, 2, axis=axid)
    pathr, pathc = map(np.ravel, [pathr, pathc])

    # TODO: add path width

    bpm = bsvol[pathc, pathr, :]

    return bpm


class BoundedArea(object):
    """
    Bounded area object from a parametric equation.
    """

    def __init__(self, image=None, shape=None, subimage=None):
        """ Initialization of the

        :Parameters:
            image : 2d array
                Image to generate mask.
            shape : tuple/list
                Shape of the image matrix.
            subimage : 2d bool array
                Image generated.
        """

        self.image = image
        if rc is None:
            self.row, self.col = image.shape
        else:
            self.row, self.col = shape
        self.rgrid, self.cgrid = np.meshgrid(range(self.row), range(self.col))

        # Subimage comprises of the image segment within the overall image
        if subimage is None:
            self.subimage = self.image.copy()
        else:
            self.subimage = subimage

    @property
    def mask(self):
        """ Subimage attribute as mask
        """

        return self.subimage.astype('bool')

    @property
    def subgrid(self):
        """ Substituent pixel coordinates of the image.
        """

        sg = np.stack(np.where(self.subimage == 1))
        return sg

    # Logical operations between BoundedArea instances
    def __and__(self, other):
        """ Logical and operation.
        """

        subimage_and = self.mask & other.mask

        return BoundedArea(image=self.image, subimage=subimage_and)

    def __or__(self, other):
        """ Logical or operation.
        """

        subimage_or = self.mask | other.mask

        return BoundedArea(image=self.image, subimage=subimage_or)

    def __invert__(self):
        """ Logical invert operation
        """

        subimage_inv = ~ self.subimage

        return BoundedArea(image=self.image, subimage=subimage_inv)

    def setBoundary(self, pmz='linear', boundtype='>', **kwds):
        """ Add bound to grid to redefine subgrid.

        :Parameters:
            pmz : str | 'linear'
                Parametrization (pmz) of the decision boundary ('linear' or 'circular').
            boundtype : str | '>'
                Bound region specification ('>' or '<').
            **kwds : keyword arguments
        """

        if pmz == 'linear':

            # Construct decision boundary y = kx + b or r = kc + b based on two points
            pa, pb = kwds.pop('points') # Points follow (row, column) index convention
            k = (pb[1] - pa[1]) / (pb[0] - pa[0])
            b = pa[1] - k * pa[0]

            if boundtype == '>': # Keep the upper end
                self.subrgrid, self.subcgrid = np.where(self.rgrid > k * self.cgrid + b)

            elif boundtype == '<': # Keep the lower end
                self.subrgrid, self.subcgrid = np.where(self.rgrid < k * self.cgrid + b)

            self.subimage = aly._signedmask(self.row, self.col, self.subrgrid, self.subcgrid, sign=1)

        elif pmz == 'circular':

            # Construct decision boundary (r-r0)^2 + (c-c0)^2 = 1 based on center and radius
            pc = kwds.pop('center') # in (row, column) format
            rad = kwds.pop('radius')

            if boundtype == '>': # Select inner circle
                self.subimage, _, region = \
                aly.circmask(self.image, pc[0], pc[1], rad, sign=0, ret='all', **kwds)
                self.subrgrid, self.subcgrid = region

            elif boundtype == '<': # Select outer circle
                self.subimage, _, region = \
                aly.circmask(self.image, pc[0], pc[1], rad, sign=1, ret='all', **kwds)
                self.subrgrid, self.subcgrid = region

        else:
            raise NotImplementedError

    def view(self, origin='lower', cmap='terrain_r', axes=True, **kwds):
        """ Display the current mask.

        :Parameters:
            origin : str | 'lower'
                Location of the image origin.
            cmap : str | 'terrain_r'
                Color map
            axes : bool | True
                Axes visibility option in plot.
            **kwds : keyword arguments
                Additional arguments for `matplotlib.pyplot.imshow()`.
        """

        f, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(self.subimage, origin=origin, cmap=cmap, **kwds)

        if axes == False:
            ax.set_xticks([])
            ax.set_yticks([])

    def toMask(self, inbound=1, exbound=0):
        """ Generate a scaled mask from existing shape.

        :Parameters:
            inbound : float | 1
                Value for the pixels within the boundary.
            exbound : float | 0
                Value for the pixels outside the boundary.

        :Return:
            modmask : 2d array
                Modified mask as a 2d array.
        """

        modmask = self.subimage.copy()
        modmask[modmask==1] = inbound
        modmask[modmask==0] = exbound

        return modmask


# ================ #
# Image correction #
# ================ #

def fitEllipseParams(*coords, plot=False, img=None, **kwds):
    """
    Direct least-squares method for fitting ellipse from scattered points
    """

    rcoords, ccoords = coords
    fitvec = elf.fitEllipse(rcoords, ccoords)

    # Calculate the ellipse parameters
    center = elf.ellipse_center(fitvec)
    phi = elf.ellipse_angle_of_rotation(fitvec)
    axes = elf.ellipse_axis_length(fitvec)

    if plot:    # Generate a diagnostic plot of the fitting result
        a, b = axes
        R = np.arange(0, 2*np.pi, 0.01)
        x = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
        y = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)

        fsize = kwds.pop('figsize', (6, 6))
        f, ax = plt.subplots(figsize=fsize)
        try:
            ax.imshow(img, origin='lower', cmap='terrain_r')
        except:
            raise ValueError('Need to supply an image for plotting!')
        ax.scatter(rcoords, ccoords, 15, 'k')
        ax.plot(x, y, color = 'red')

    return center, phi, axes


def vertexGenerator(center, fixedvertex, arot, direction=-1, scale=1, ret='all'):
    """
    Generation of the vertices of symmetric polygons.

    :Parameters:
        center : (int, int)
            Pixel positions of the symmetry center (row pixel, column pixel).
        fixedvertex : (int, int)
            Pixel position of the fixed vertex (row pixel, column pixel).
        arot : float
            Spacing in angle of rotation.
        direction : int | 1
            Direction of angular rotation (1 = anticlockwise, -1 = clockwise)
        scale : float
            Radial scaling factor.
        ret : str | 'all'
            Return type. Specify 'all' returns all vertices, specify 'generated'
            returns only the generated ones (without the fixedvertex in the argument).

    :Return:
        vertices : 2D array
            Collection of generated vertices.
    """

    if type(arot) in (int, float):
        nangles = int(np.round(360 / arot)) - 1 # Number of angles needed
        rotangles = direction*np.linspace(1, nangles, nangles)*arot
    else:
        nangles = len(arot)
        rotangles = np.cumsum(arot)

    # Reformat the input array to satisfy function requirement
    fixedvertex_reformatted = np.array(fixedvertex, dtype='int32', ndmin=2)[None,...]

    if ret == 'all':
        vertices = [fixedvertex]
    elif ret == 'generated':
        vertices = []

    if type(scale) in (int, float):
        scale = np.ones((nangles,)) * scale

    # Generate reference points by rotation and scaling
    for ira, ra in enumerate(rotangles):

        rmat = cv2.getRotationMatrix2D(center, ra, scale[ira])
        rotvertex = np.squeeze(cv2.transform(fixedvertex_reformatted, rmat)).tolist()
        vertices.append(rotvertex)

    return np.asarray(vertices, dtype='int32')


def affineWarping(img, landmarks, refs, ret='image'):
    """
    Perform image warping based on a generic affine transform (homography).

    :Parameters:
        img : 2D array
            Input image (distorted)
        landmarks : list/array
            List of pixel positions of the
        refs : list/array
            List of pixel positions of regular

    :Returns:
        imgaw : 2D array
            Image after affine warping.
        maw : 2D array
            Homography matrix for the tranform.
    """

    landmarks = np.asarray(landmarks, dtype='float32')
    refs = np.asarray(refs, dtype='float32')

    maw, _ = cv2.findHomography(landmarks, refs)
    imgaw = cv2.warpPerspective(img, maw, img.shape)

    if ret == 'image':
        return imgaw
    elif ret == 'all':
        return imgaw, maw


def applyWarping(imgstack, axis, hgmat):
    """
    Apply warping transform for a stack of images along an axis

    :Parameters:
        imgstack : 3D array
            Image stack before warping correction.
        axis : int
            Axis to iterate over to apply the transform.
        hgmat : 2D array
            Homography matrix.

    :Return:
        imstack_transformed : 3D array
            Stack of images after correction for warping.
    """

    imgstack = np.moveaxis(imgstack, axis, 0)
    imgstack_transformed = np.zeros_like(imgstack)
    nimg = imgstack.shape[0]

    for i in range(nimg):
        img = imgstack[i,...]
        imgstack_transformed[i,...] = cv2.warpPerspective(img, hgmat, img.shape)

    imgstack_transformed = np.moveaxis(imgstack_transformed, 0, axis)

    return imgstack_transformed


class MomentumCorrector(object):
    """
    Momentum distortion correction and momentum calibration workflow.
    """

    def __init__(self, image, rotsym=6):
        """
        :Parameters:
            image : 3d array
                Volumetric band structure data.
            rotsym : int | 6
                Order of rotational symmetry.
        """

        self.image = np.squeeze(image)
        self.imgndim = image.ndim
        if (self.imgndim > 3) or (self.imgndim < 2):
            raise ValueError('The input image dimension need to be 2 or 3!')

        self.rotsym = int(rotsym)
        self.rotsym_angle = int(360 / self.rotsym)
        self.arot = np.array([0] + [self.rotsym_angle]*(self.rotsym-1))
        self.ascale = np.array([1.0]*self.rotsym)
        self.features = {}

    def selectSlice2D(self, selector, axis=2):
        """ Select (hyper)slice from a (hyper)volume.

        :Parameters:
            selector : slice object
                Selector for the slice (image).
            axis : int | 2
                Axis along which to select the image.
        """

        if self.imgndim > 2:
            im = np.moveaxis(self.image, axis, 0)
            self.slice = im[selector,...].sum(axis=0)
        elif self.imgndim == 2:
            raise ValueError('Input image dimension is already 2!')

    def featureExtract(self, image, direction='ccw', type='points', center_det='centroidnn', **kwds):
        """ Extract features from the selected (hyper)slice.

        :Parameters:
            image : 2d array
                Image slice to extract features from.
            direction : str | 'ccw'
                Circular direction to reorder the features in ('cw' or 'ccw').
            type : str | 'points'
                Type of features to extract.
            **kwds : keyword arguments
                extra arguments for `symmetrize.pointops.peakdetect2d()`.
        """

        if type == 'points':

            self.center_detection_method = center_det
            # Detection and ordering of geometric landmarks
            self.peaks = po.peakdetect2d(image, **kwds)
            self.pcent, self.pouter = po.pointset_center(self.peaks, method=center_det, ret='cnc')
            self.pcent = tuple(self.pcent)
            self.pouter_ord = po.pointset_order(self.pouter, direction=direction)
            # Construct feature dictionary
            self.features['verts'] = self.pouter_ord
            self.features['center'] = np.atleast_2d(self.pcent)
            # Calculate geometric distances
            self.mcvdist = po.cvdist(self.pouter_ord, self.pcent).mean()
            self.mvvdist = po.vvdist(self.pouter_ord).mean()

            if self.rotsym == 6:
                self.mdist = (self.mcvdist + self.mvvdist) / 2
                self.mcvdist = self.mdist
                self.mvvdist = self.mdist

        else:
            raise NotImplementedError

    def _featureUpdate(self, center_det='centroidnn', **kwds):
        """ Update selected features.
        """

        image = kwds.pop('image', self.slice)
        # Update the point landmarks in the transformed coordinate system
        pks = po.peakdetect2d(image, **kwds)
        self.pcent, self.pouter = po.pointset_center(pks, method=center_det)
        self.pouter_ord = po.pointset_order(self.pouter, direction='ccw')
        self.pcent = tuple(self.pcent)
        self.features['verts'] = self.pouter_ord
        self.features['center'] = np.atleast_2d(self.pcent)

    def _imageUpdate(self):
        """ Update distortion-corrected images.
        """

        try:
            self.slice = self.slice_corrected
            del self.slice_corrected
        except:
            pass

        try:
            self.image = self.image_corrected
            del self.image_corrected
        except:
            pass

    def update(self, content, **kwds):
        """ Update specific attributes of the class.

        :Parameters:
            content : str | 'all'
                'feature' = update only feature attributes
                'image' = update only image-related attributes
                'all' = update both feature and image-related attributes
            **kwds : keyword arguments
                Extra arguments passed into `self._featureUpdate()`
        """

        if content == 'feature':
            self._featureUpdate(**kwds)
        elif content == 'image':
            self._imageUpdate()
        elif content == 'all':
            self._imageUpdate()
            self._featureUpdate(**kwds) # Feature update comes after image update

    def linWarpEstimate(self, weights=(1, 1, 1), niter=50, method='Nelder-Mead',
                        ret=True, **kwds):
        """ Estimate the linear deformation field.

        :Parameters:
            weights : tuple/list/array
                Weights added to the terms in the optimizer.
            niter : int | 50
                Maximum number of iterations.
            method : str | 'Nelder-Mead'
                Name of the optimization method.
            ret : bool | True
                Specify if returning the corrected image slice.
            **kwds : keyword arguments
            ========= ========== =============================================
            keyword   data type  meaning
            ========= ========== =============================================
            landmarks list/array Symmetry landmarks selected for registration
            fitinit   tuple/list Initial conditions for fitting
            ========= ========== =============================================
        """

        landmarks = kwds.pop('landmarks', self.pouter_ord)
        # Set up the initial condition for the optimization for symmetrization
        fitinit = np.asarray([self.arot, self.ascale]).ravel()
        self.init = kwds.pop('fitinit', fitinit)

        self.prefs, _ = sym.refsetopt(self.init, landmarks, self.pcent, self.mcvdist, self.mvvdist, niter=niter,\
                                        direction=1, weights=weights, method=method, stepsize=0.5)

        # Calculate linearly warped image and landmark positions
        self.slice_corrected, self.linwarp = sym.imgWarping(self.slice, landmarks=landmarks, refs=self.prefs)

        if ret:
            return self.slice_corrected

    @staticmethod
    def transform(points, transmat):
        """ Coordinate transform of a point set in the (row, column) formulation

        :Parameters:
            points : list/array
                Cartesian pixel coordinates of the points to be transformed.
            transmat : 2D array
                The transform matrix.

        :Return:
            Transformed point coordinates.
        """

        pts_cart_trans = sym.pointsetTransform(np.roll(points, shift=1, axis=1), transmat)

        return np.roll(pts_cart_trans, shift=1, axis=1)

    def nonlinWarpEstimate(self, image, axis, rand_amp=1, ret=True):
        """ Estimate the nonlinear deformation field using thin plate spline.
        """

        self.prefs = sym.vertexGenerator(self.pcent, self.pouter_ord[0,:], self.arot, direction=-1,
                                         scale=self.ascale, rand_amp=rand_amp, ret='all')[1:,:]
        self.image_corrected, self.nonlinwarp = tps.tpsWarping(self.pouter_ord, self.prefs, image, axis)

        if ret:
            return self.image_corrected

    def rotate(self, angle='auto', ret=False, **kwds):
        """ Rotate 2D image in the homogeneous coordinate.

        :Parameters:
            angle : float/str
                Angle of rotation (specify 'auto' to use automated estimation).
            ret : bool | False
                Return specification (True/False)
            **kwds : keyword arguments
            ======= ========== =======================================
            keyword data type  meaning
            ======= ========== =======================================
            image   2d array   2D image for correction
            center  tuple/list pixel coordinates of the image center
            scale   float      scaling factor in rotation
            ======= ========== =======================================
            See `symmetrize.sym.sym_pose_estimate()` for other keywords.
        """

        image = kwds.pop('image', self.slice)
        center = kwds.pop('center', self.pcent)
        scale = kwds.pop('scale', 1)

        if angle == 'auto':
            center = tuple(np.asarray(center).astype('int'))
            angle_auto, _ = sym.sym_pose_estimate(image/image.max(), center, **kwds)
            self.image_rot, rotmat = _rotate2d(image, center, angle_auto, scale)
        else:
            self.image_rot, rotmat = _rotate2d(image, center, angle, scale)

        # Compose the rotation matrix with the previously determined warping matrix
        self.composite_linwarp = np.dot(rotmat, self.linwarp)

        if ret:
            return rotmat

    def correct(self, axis, use_composite_transform=False, update=False, **kwds):
        """ Apply a 2D transform to a stack of 2D images (3D).

        :Parameters:
            axis : int
                Axis for slice selection.
            use_composite_transform : bool | False
                Option to use the composite transform involving the rotation.
            update : bool | False
                Option to update the existing figure attributes.
            **kwds : keyword arguments
            ======= ========= ===============================
            keyword data type meaning
            ======= ========= ===============================
            image   2d array  3D image for correction
            warping 2d array  2D transform correction matrix
            ======= ========= ===============================
        """

        image = kwds.pop('image', self.image)
        if use_composite_transform == True:
            hgmat = kwds.pop('warping', self.composite_linwarp)
        else:
            hgmat = kwds.pop('warping', self.linwarp)

        self.image_corrected = sym.applyWarping(image, axis, hgmat=hgmat)

        if update != False:
            if update == True:
                self.update('all')
            else:
                self.update(update)

    @staticmethod
    def getWarpFunction(**kwds):
        """ Construct warping function to apply to other datasets.
        """

        warping = kwds.pop('warping', np.eye(3))
        warpfunc = partial(base.correctnd, warping=warping)

        return warpfunc

    def view(self, origin='lower', cmap='terrain_r', figsize=(4, 4), points={},
             annotated=False, ret=False, imkwd={}, **kwds):
        """ Generate imshow plot.

        :Parameters:
            origin : str | 'lower'
                Figure origin specification ('lower' or 'upper').
            cmap : str | 'terrain_r'
                Colormap specification.
            figsize : tuple/list | (4, 4)
                Figure size.
            points : dict | {}
                Points for annotation.
            annotated : bool | False
                Option for annotation.
            ret : bool | False
                Option to return figure and axis objects.
            imkwd : dict | {}
                Keyword arguments for matplotlib.pyplot.imshow().
            **kwds : keyword arguments
                General extra arguments for the plotting procedure.
        """

        image = kwds.pop('image', self.slice)
        f, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, origin=origin, cmap=cmap, **imkwd)

        # Add annotation to the figure
        if annotated:
            tsr, tsc = kwds.pop('textshift', (3, 3))
            txtsize = kwds.pop('textsize', 12)

            for pk, pvs in points.items():
                try:
                    ax.scatter(pvs[:,1], pvs[:,0])
                except:
                    ax.scatter(pvs[1], pvs[0])

                if pvs.size > 2:
                    for ipv, pv in enumerate(pvs):
                        ax.text(pv[1]+tsc, pv[0]+tsr, str(ipv), fontsize=txtsize)

        if ret:
            return f, ax

    def calibrate(self, image, point_from, point_to, dist, ret='coeffs'):
        """ Calibration of the momentum axes. Obtain all calibration-related values,
        return only the ones requested.

        :Parameters:
            image : 2d array
                Image slice to construct the calibration function.
            point_from, point_to : list/tuple, list/tuple
                Pixel coordinates of the two special points.
            dist : float
                Distance between the two selected points in inverse Angstrom.
            ret : str | 'coeffs'
                Specification of return values ('axes', 'extent', 'coeffs', 'grid', 'func', 'all').
        """

        self.calibration = calibrateK(image, point_from, point_to, dist, ret='all')

        if ret != False:
            try:
                return project(self.calibration, [ret])
            except:
                return project(self.calibration, ret)

    def saveImage(self, form='tiff', save_addr='./', dtyp='float32', **kwds):
        """ Save the distortion-corrected dataset (image only, without axes).

        :Parameters:
            form : str | 'tiff'
                File format for saving the corrected image ('tiff' or 'mat').
            save_addr : str | './'
                The address to save the file at.
            dtyp : str | 'float32'
                Data type (in case conversion if needed).
            **kwds : keyword arguments
                See keywords from `tifffile.imsave()`.
        """

        data = kwds.pop('data', self.image).astype(dtyp)
        save_addr = u.appendformat(save_addr, form)

        if form == 'tiff':

            try:
                import tifffile as ti
                ti.imsave(save_addr, data=data, **kwds)

            except ImportError:
                raise('tifffile package is not installed locally!')

        elif form == 'mat':

            sio.savemat(save_addr, {'data':data})

    def saveParameters(self, form='h5', save_addr='./momentum'):
        """
        Save all the attributes of the workflow instance for later use
        (e.g. momentum scale conversion, reconstructing the warping map function).

        :Parameters:
            form : str | 'h5'
                File format to for saving the parameters ('h5'/'hdf5', 'mat')
            save_addr : str | './momentum'
                The address for the to be saved file.
        """

        base.saveClassAttributes(self, form, save_addr)


def _rotate2d(image, center, angle, scale=1):
    """
    2D matrix rotation.

    :Parameters:
        image : 2d array
        center : tuple/list
        angle : numeric
        scale : numeric | 1
    """

    rotmat = cv2.getRotationMatrix2D(center, angle=angle, scale=scale)
    # Construct rotation matrix in homogeneous coordinate
    rotmat = np.concatenate((rotmat, np.array([0, 0, 1], ndmin=2)), axis=0)

    image_rot = cv2.warpPerspective(image, rotmat, image.shape)

    return image_rot, rotmat


# ================ #
# Fitting routines #
# ================ #

SQ2 = np.sqrt(2.0)
SQ2PI = np.sqrt(2*np.pi)


def gaussian(feval=False, vardict=None):
    """Gaussian model
    """

    asvars = ['amp', 'xvar', 'ctr', 'sig']
    expr = 'amp*np.exp(-((xvar-ctr)**2) / (2*sig**2))'

    if feval == False:
        return asvars, expr
    else:
        return eval(expr, vardict, globals())


def voigt(feval=False, vardict=None):
    """Voigt model
    """

    asvars = ['amp', 'xvar', 'ctr', 'sig', 'gam']
    expr = 'amp*wofz((xvar-ctr+1j*gam) / (sig*SQ2)).real / (sig*SQ2PI)'

    if feval == False:
        return asvars, expr
    else:
        return eval(expr, vardict, globals())


def func_update(func, suffix=''):
    """
    Attach a suffix to parameter names and their instances
    in the expression of a function

    ***Parameters***

    func : function
        input function
    suffix : str | ''
        suffix to attach to parameter names

    ***Returns***

    params : list of str
        updated function parameters
    expr : str
        updated function expression
    """

    _params, _expr = func(feval=False)

    # Update function parameter list
    params = list(map(lambda p: p + suffix, _params))

    # Update function expression string
    replacements = np.array([_params, params]).T.tolist()
    expr = reduce(lambda string, parampairs: string.replace(*parampairs), replacements, _expr)

    return params, expr


def func_add(*funcs):
    """
    Addition of an arbitray number of functions

    ***Parameters***

    *funcs : list/tuple
        functions to combine

    ***Returns***

    funcsum : function
        functional sum
    """

    # Update the function variables with suffixes
    fparts = np.asarray([func_update(f, str(i)) for i, f in enumerate(funcs)]).T.tolist()

    # Generate combined list of variables and expression string
    asvars = reduce(op.add, fparts[0])
    expr = reduce(op.add, map(lambda x: x+' + ', fparts[1]))[:-3]

    def funcsum(feval=False, vardict=None):

        if feval == False:
            return asvars, expr
        else:
            try:
                return eval(expr, vardict, globals())
            except:
                raise Exception('Not all variables can be assigned.')

    return funcsum


def bootstrapfit(data, axval, model, params, axis=0, dfcontainer=None, pbar=False, **kwds):
    """
    Line-by-line fitting via bootstrapping fitted parameters from one line to the next

    :Parameters:
        data : ndarray
            Data used in fitting.
        axval : list/numeric array
            Value for the axis.
        model : lmfit Model object
            The fitting model.
        params : lmfit Parameters object
            Initial guesses for fitting parameters.
        axis : int | 0
            The axis of the data to fit.
        dfcontainer : pandas DataFrame | None
            Dataframe container for the fitting parameters.
        pbar : bool | False
            Progress bar condition.
        **kwds : keyword arguments
            =============  ==========  ===================================
            keyword        data type   meaning
            =============  ==========  ===================================
            maxiter        int         maximum iteration per fit (default = 20)
            concat         bool        concatenate the fit parameters to DataFrame input
                                       False (default) = no concatenation, use an empty DataFrame to start
                                       True = with concatenation to input DataFrame
            bgremove       bool        toggle for background removal (default = True)
            flipped        bool        toggle for fitting start position
                                       (if flipped, fitting start from the last line)
            verbose        bool        toggle for output message (default = False)
            =============  ==========  ===================================

    :Returns:
        df_fit : pandas DataFrame
            Dataframe container populated with obtained fitting parameters.
        data_nobg : ndarray
            Background-removed (Shirley-type) traces.
    """

    # Retrieve values from input arguments
    vb = kwds.pop('verbose', False)
    maxiter = kwds.pop('maxiter', 20)
    concat = kwds.pop('concat', False)
    bgremove = kwds.pop('bgremove', True)
    cond_flip = int(kwds.pop('flipped', False))

    data = np.rollaxis(data, axis)
    # Flip axis if the conditional is True
    data = cond_flip*np.flip(data, axis=0) + (1-cond_flip)*data
    nr, nc = data.shape

    # Save background-removed data
    data_nobg = np.zeros_like(data)

    # Construct container for fitting parameters
    if dfcontainer is None:
        df_fit = pd.DataFrame(columns=params.keys())
    elif isinstance(dfcontainer, pd.core.frame.DataFrame):
        dfcontainer.sort_index(axis=1, inplace=True)
        if concat == False:
            df_fit = dfcontainer[0:0]
        else:
            df_fit = dfcontainer
    else:
        raise Exception('Input dfcontainer needs to be a pandas DataFrame!')

    # Fitting every line in data matrix
    for i in tqdm(range(nr), disable=not(pbar)):

        # Remove Shirley background (nobg = no background)
        line = data[i,:]
        if bgremove == True:
            sbg = shirley(axval, line, maxiter=maxiter, warning=False, **kwds)
            line_nobg = line - sbg
        else:
            line_nobg = line
        data_nobg[i,:] = line_nobg
        out = model.fit(line_nobg, params, x=axval)

        # Unpacking dictionary
        currdict = {}
        for _, param in out.params.items():
            currdict[param.name] = param.value
            currdf = pd.DataFrame.from_dict(currdict, orient='index').T

        df_fit = pd.concat([df_fit, currdf], ignore_index=True, sort=True)

        # Set the next fit initial guesses to be
        # the best values from the current fit
        bestdict = out.best_values
        for (k, v) in bestdict.items():
            params[k].set(v)

        if vb == True:
            print("Finished line {}/{}...".format(i+1, nr))

    # Flip the rows if fitting is conducted in the reverse direction
    if cond_flip == 1:
        df_fit = df_fit.iloc[::-1]
        df_fit.reset_index(drop=True, inplace=True)
        data_nobg = np.flip(data_nobg, axis=0)

    return df_fit, data_nobg


class Model(object):
    """
    Class of fitting curve models
    """

    def __init__(self, func, xvar, name=None):
        self.func = func
        self.params, self.expr = func(feval=False)
        if name is None and hasattr(self.func, '__name__'):
            name = self.func.__name__
        self.name = name
        self.xvar = xvar


    def __repr__(self):
        return '<{}.{}: {}>'.format(self.__module__, \
                self.__class__.__name__, self.name)


    @staticmethod
    def normalize(data):
        """
        Normalize n-dimensional data
        """
        return data/np.max(np.abs(data))


    def model_eval(self, params):
        """
        Evaluate the fitting model with given parameters
        """
        return self.func(feval=True, vals=params, xvar=self.xvar)


    def partial_eval(self, params, part=0):
        """
        Evaluate parts of a composite fitting model
        """
        pass


    def _costfunc(self, inits, xv, form='original'):
        """
        Define the cost function of the optimization process
        """
        self.model = self.func(feval=True, vals=inits, xvar=xv)
        if form == 'original':
            cf = self.data - self.model
        elif form == 'norm':
            cf = self.norm_data - self.model

        return cf.ravel()


    def fit(self, data, inits, method='leastsq', **fitkwds):
        """
        Run the optimization
        """
        self.data = data
        self.norm_data = self.normalize(data)
        self.inits = inits

        if method == 'leastsq':
            fitout = opt.leastsq(self._costfunc, self.inits, args=self.xvar, \
            xtol=1e-8, gtol=1e-6, full_output=True, **fitkwds)
        elif 'minimize' in method:
            method_str = method.split('_')[1]
            fitout = opt.minimize(self._costfunc, self.inits, args=self.xvar,\
            method=method_str, **fitkwds)

        return fitout


#====================================#
# Fitting result parsing and testing #
#====================================#

def build_dynamic_matrix(fitparams, display_range=slice(None, None, None), pre_t0_range=slice(None, 1, None)):
    """
    Construct the dynamic matrix from the fitting results:
    for each fitting parameter, construct time-dependent value,
    time-dependent absolute and relative changes

    ***Parameters***

    fitparams : 3D ndarray
        fitting output
    display_range : slice object | slice(None, None, None)
        display time range of the fitting parameters (default = full range)
    pre_t0_range : slice object | slice(None, 1, None)
        time range regarded as before time-zero

    ***Returns***

    dyn_matrix : 4D ndarray
        calculated dynamic matrix
    """

    if np.ndim(fitparams) != 3:
        raise Exception('Fitting results input need to be a 3D array!')
    else:
        nt, nparam, nk = fitparams.shape
        ncol = 3
        ndisp = len(range(*display_range.indices(nt))) # length of remaining time points
        reduced_fitparams = fitparams[display_range,...]
        dyn_matrix = np.zeros((nk, ndisp, nparam, ncol))

        # Fill the dynamic matrix by values from each change parameters
        for idx, i in enumerate(range(nparam)):

            # Calculate the k-dependent pre-t0 values
            I0 = np.mean(fitparams[pre_t0_range, i, :], axis=0)

            # Calculate the k-dependent absolute and relative changes
            dyn_matrix[..., idx, 0] = np.transpose(reduced_fitparams[:, i, :])
            dyn_matrix[..., idx, 1] = np.transpose(reduced_fitparams[:, i, :] - I0)
            dyn_matrix[..., idx, 2] = np.transpose((reduced_fitparams[:, i, :] - I0) / I0)

        return dyn_matrix
