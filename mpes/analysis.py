#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""
# =======================================
# Sections:
# 1.  Utility functions
# 2.  Background removal
# 3.  Image segmentation
# 4.  Fitting routines
# 5.  Fitting result parsing and testing
# =======================================

from __future__ import print_function, division
from . import utils as u
import numpy as np
from numpy.linalg import norm
import pandas as pd
from skimage import measure, filters, morphology
from skimage.draw import circle, polygon
from math import cos, pi
import scipy.optimize as opt
from scipy.special import wofz
from functools import reduce
import operator as op


# =================== #
#  Utility functions  #
# =================== #

def sortByAxes(arr, axes):
    """
    Sort n-dimensional array into ascending order
    based on the corresponding axes order

    **Parameters**

    arr : numeric nD array
        the nD array to be sorted
    axes : tuple/list
        list of axes

    **Return**
    if no sorting is needed, returns None

    if the ndarray and axes are sorted,
    return the sorted values
    """

    arr = np.asarray(arr)
    dim = np.ndim(arr)
    dimax = len(axes)
    if dim != dimax:
        raise Exception('The number of axes should match the dimenison of arr!')

    sortseq = np.zeros(dim)
    # Sort the axes vectors in ascending order
    if dimax == 1:
        sortedaxes = np.sort(axes)
        if np.prod(sortedaxes == axes) == 1:
            seq = 0
        elif np.prod(sortedaxes == axes[::-1]) == 1:
            seq = 1
            arr = arr[::-1]
        sortseq[0] = seq
    else:
        sortedaxes = list(map(np.sort, axes))

        # Check which axis changed, sort the array accordingly
        for i in range(dim):

            seq = None
            # if an axis is in ascending order
            if np.prod(sortedaxes[i] == axes[i]) == 1:
                seq = 0

            # if an axis is in descending order
            elif np.prod(sortedaxes[i] == axes[i][::-1]) == 1:
                seq = 1
                arr = u.revaxis(arr, axis=i)

            sortseq[i] = seq

    # Return sorted arrays or None if sorting is not needed
    if np.any(sortseq == 1):
        return arr, sortedaxes
    else:
        return


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
    tol : float
        fitting tolerance
    maxiter : int | 20
        maximal iteration
    explicit : bool | False
        explicit display of iteration number
    warning : bool | True
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


def peakdetect(y_axis, x_axis = None, lookahead = 200, delta=0):
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


    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
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


    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError: # When no peaks have been found
        pass

    return [max_peaks, min_peaks]


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

    :Return:
        cmask or cmask*img : 2D array
            Mask only or masked image
    """

    rim, cim = img.shape
    shape = kwds.pop('shape', (rim, cim))

    # Generate circular mask of the chosen sign
    rr, cc = circle(rcent, ccent, rad, shape=shape)
    cmask = _signedmask(rim, cim, rr, cc, sign=sign)

    if ret == 'mask':
        return cmask
    elif ret == 'masked_image':
        return cmask*img


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
        kwds : keyword arguments

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


def apply_mask_along(arr, mask, axes=None):
    """
    Apply a mask in a low dimensional slice throughout a high-dimensional array

    :Parameters:
        arr : nD array
            Multidimensional array for masking.
        mask : nD array
            Mask to apply.
        axes : list/tuple of int
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

    return maskedarr


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


def bootstrapfit(data, axval, model, params, axis=0, dfcontainer=None, **kwds):
    """
    Line-by-line fitting via bootstrapping fitted parameters from one line to the next

    ***Parameters***

    data : ndarray
        data used in fitting
    axval : list/numeric array
        value for the axis
    model : lmfit Model object
        fitting model
    params : lmfit Parameters object
        initial guesses for fitting parameters
    axis : int | 0
        axis of the data to fit
    dfcontainer : pandas DataFrame | None
        container for the fitting parameters
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

    ***Returns***

    df_fit : pandas DataFrame
        filled container for fitting parameters
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
    for i in range(nr):

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

        df_fit = pd.concat([df_fit, currdf], ignore_index=True)

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
