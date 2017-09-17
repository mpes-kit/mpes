#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from __future__ import print_function, division
from . import utils as u
import numpy as np
from numpy.linalg import norm
import pandas as pd
from skimage import measure, filters, morphology
from math import cos, pi
import scipy.optimize as opt
from scipy.special import wofz
from functools import reduce
import operator as op


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
                arr = u.revaxis(arr, axis=i)
                
            sortseq[i] = seq
        
    # Return sorted arrays or None if sorting is not needed
    if np.any(sortseq == 1):
        return arr, sortedaxes
    else:
        return


#====================#
# Background removal #
#====================#

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


#====================#
# Image segmentation #
#====================#

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


#===============#
# Model fitting #
#===============#

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