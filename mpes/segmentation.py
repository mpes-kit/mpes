#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from __future__ import print_function, division
import numpy as np
from numpy.linalg import norm
import pandas as pd
from skimage import measure, filters, morphology
from math import cos, pi
from scipy.special import wofz

def to_odd(num):
    """
    Convert a single number to its nearest odd number
    
    **Parameters**
    
    num : float/int
    
    **Return**
    
    oddnum : int
        the nearest odd number
    """

    rem = round(num) % 2
    oddnum = num + int(cos(rem*pi/2))
    
    return oddnum


def revaxis(arr, axis=-1):
    """
    Reverse an ndarray along certain axis
    
    **Parameters**
    arr : nD numeric array
        array to invert
    axis : int | -1
        the axis along which to invert
    
    **Return**
    revarr : nD numeric array
        axis-inverted nD array
    """
    
    arr = np.asarray(arr).swapaxes(axis, 0)
    arr = arr[::-1,...]
    revarr = arr.swapaxes(0, axis)
    return revarr


def shirley(x, y, tol=1e-5, maxiter=20, explicit=False, warning=False):
    """
    Calculate the 1D best auto-Shirley background S for a dataset (x, y).
    Adapted from Kane O'Donnell's routine
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

    # Locate the biggest peak.
    maxidx = abs(y - np.amax(y)).argmin()

    # It's possible that maxidx will be 0 or -1. If that is the case,
    # we can't use this algorithm, we return a zero background.
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
    # and B is equal to (yl - yr) below lmidx and initially zero above.
    B = np.zeros(x.shape)
    B[:lmidx] = yl - yr
    Bnew = B.copy()

    niter = 0
    while niter < maxiter:
        if explicit:
            print("Shirley iteration: " + str(it))
        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        ksum = 0.0
        for i in range(lmidx, imax):
            ksum += (x[i] - x[i + 1]) * 0.5 * (y[i] + y[i + 1]
                                               - 2 * yr - B[i] - B[i + 1])
        k = (yl - yr) / ksum
        # Calculate new B
        for i in range(lmidx, rmidx):
            ysum = 0.0
            for j in range(i, imax):
                ysum += (x[j] - x[j + 1]) * 0.5 * (y[j] +
                                                   y[j + 1] - 2 * yr - B[j] - B[j + 1])
            Bnew[i] = k * ysum
        # If Bnew is close to B, exit.
        if norm(Bnew - B) < tol:
            B = Bnew.copy()
            break
        else:
            B = Bnew.copy()
        niter += 1

    if niter >= maxiter and warning == True:
        print("Max iterations exceeded before convergence.")

    if is_reversed:
        return (yr + B)[::-1]
    else:
        return yr + B

        
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
    dmax = to_odd(max(img.shape))
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
    if np.ndim(axes) == 1:
        dimax = 1
    else:
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
                arr = revaxis(arr, axis=i)
                
            sortseq[i] = seq
        
    # Return sorted arrays or None if sorting is not needed
    if np.any(sortseq == 1):
        return arr, sortedaxes
    else:
        return

#################
# Model fitting #
#################
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
        verbose        bool        toggle for output message (default = False)
        =============  ==========  ===================================
    
    ***Returns***
    
    df_fit : pandas DataFrame
        filled container for fitting parameters
    """
    
    # Retrieve values from input arguments
    data = np.rollaxis(data, axis)
    nr, nc = data.shape
    vb = kwds.pop('verbose', False)
    maxiter = kwds.pop('maxiter', 20)
    concat = kwds.pop('concat', False)
    
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
        
        line = data[i,:]
        # Remove shirley background
        sbg = shirley(axval, line, maxiter=maxiter, warning=False, **kwds)
        line_nobg = line - sbg
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
            print("Finished {}/{}...".format(i+1, nr))
    
    return df_fit