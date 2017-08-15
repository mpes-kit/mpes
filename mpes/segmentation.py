#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from __future__ import print_function, division
import numpy as np
from numpy.linalg import norm
import pandas as pd
from math import cos, pi
from skimage import measure, filters, morphology

def shirley(x, y, tol=1e-5, maxit=10):
    """
    S = shirley(x,y, tol=1e-5, maxit=10)
    Calculate the best auto-Shirley background S for a dataset (x,y). Finds the biggest peak
    and then uses the minimum value either side of this peak as the terminal points of the
    Shirley background.
    The tolerance sets the convergence criterion, maxit sets the maximum number
    of iterations.
    
    Adapted from Kane O'Donnell's routine
    """
    
    DEBUG = False

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

    it = 0
    while it < maxit:
        if DEBUG:
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
        it += 1

    if it >= maxit:
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
    
    img : numeric 2D array
        the 2D matrix to segment
    nbands : int
        number of electronic bands
    **kwds : keyword arguments

    **Return**
    
    imglabeled : labeled mask
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


def to_odd(num):
    """
    Convert to nearest odd number
    """

    rem = np.round(num) % 2
    return num + int(cos(rem*pi/2))
    
 
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
    
    ridges : list of dataframes of the ridge coordinates
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