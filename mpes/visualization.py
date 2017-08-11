#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as matgrid
from mayavi import mlab
import re


def initmpl():
    """
    Initialize mpes plot style
    """

    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{helvet}']
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['savefig.dpi'] = 100
    mpl.rcParams['xtick.major.pad'] = 5
    mpl.rcParams['ytick.major.pad'] = 5
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['xtick.major.size'] = 8
    mpl.rcParams['xtick.minor.size'] = 6
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.minor.width'] = 2
    mpl.rcParams['ytick.major.size'] = 8
    mpl.rcParams['ytick.minor.size'] = 6
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15


#==========#
# 1D plots #
#==========#

def stackedlineplot(datamat, axis=0, interval=0, binning=1, **kwds):
    """
    Stacked line plots (used for visualizing energy or momentum dispersion curves)
    
    **Parameters**
    
    data : numeric 2D array
        the 2D data to plot
    axis : int | 0
        the axis to cut along
    interval : float | 0
        the interval between plots
    binning : int | 1 (no binning)
        number of binned rows/columns
    **kwds : keyword arguments
        =============  ==========  ===================================
        keyword        data type   meaning
        =============  ==========  ===================================
        figsize        tuple/list  (horizontal_size, vertical_size)
        x              1D array    x axis values
        xlabel         str         x axis label
        ylabel         str         y axis label
        cmap           str         `matplotlib colormap string <https://matplotlib.org/users/colormaps.html>`_
        axislabelsize  int         font size of axis text labels
        ticklabelsize  int         font size of axis tick labels
        margins        tuple/list  (xmargin, ymargin), values between 0 and 1
        =============  ==========  ===================================
    **Return**
    
    ax : axes object
        handle for the plot axes
    """
    
    # Check binning input
    binning = round(binning)
    if binning < 1:
        binning = 1
    
    # Determine figure size
    figuresize = kwds.pop('figsize', '')
    try:
        fw, fh = vis.numFormatConversion(figuresize)
    except:
        fw, fh = 2 * figaspect(datamat)
    f, ax = plt.subplots(figsize=(fw, fh))
    
    datamat = np.rollaxis(np.asarray(datamat), axis)
    nr, nc = datamat.shape
    x = kwds.pop('x', range(0, nc))
    y = range(0, nr, binning)
    ny = len(y)
    colormap = kwds.pop('cmap', '')
    lw = kwds.pop('linewidth', 2)
    
    # Plot lines
    for ind, i in enumerate(y):
        
        yval = np.mean(datamat[i:i+binning,:], axis=0)
        line = ax.plot(x, yval + i*interval, linewidth=lw)
        # Set line color
        if colormap:
            line[0].set_color(eval('plt.cm.' + colormap + '(ind/ny)'))
    
    xlabel = kwds.pop('xlabel', '')
    ylabel = kwds.pop('ylabel', '')
    axislabelsize = kwds.pop('ax_labelsize', 12)
    ticklabelsize = kwds.pop('tk_labelsize', 10)
    
    margins = kwds.pop('margins', (0.03, 0.03))
    plt.margins(x=margins[0], y=margins[1])
    ax.set_xlabel(xlabel, fontsize=axislabelsize)
    ax.set_ylabel(ylabel, fontsize=axislabelsize)
    ax.tick_params(labelsize=ticklabelsize)
    plt.tight_layout()
    
    return ax


#==========#
# 2D plots #
#==========#

def numFormatConversion(seq, form='int', **kwds):
    """
    When length keyword is not specified as an argument, the function
    returns a format-converted sequence of numbers
    
    The function returns nothing when the conversion fails due to errors
    """
    
    try:
        lseq = len(seq)
    except:
        raise
    
    l = kwds.pop('length', lseq)
    if lseq == l:
        # Case of numeric array of the right length but may not be
        # the right type
        try:
            numseq = map(eval(form), seq)
            return numseq
        except:
            raise 
    else:
        # Case of numeric array of the right type but wrong length
        return


def colormesh2d(data, **kwds):
    """
    Efficient one-line color mesh plot of a 2D data matrix

    **Parameters**
    
    data : numeric 2D array
        the 2D data to plot
    **kwds : keyword arguments
        =============  ==========  ===================================
        keyword        data type   meaning
        =============  ==========  ===================================
        figsize        tuple/list  (horizontal_size, vertical_size)
        x              1D array    x axis coordinates
        y              1D array    y axis coordinates
        xlabel         str         x axis label
        ylabel         str         y axis label
        colormap       str         `matplotlib colormap string <https://matplotlib.org/users/colormaps.html>`_ 
        axislabelsize  int         font size of axis text labels
        ticklabelsize  int         font size of axis tick labels
        =============  ==========  ===================================

    **Return**
    
    ax : axes object
        handle for the plot axes
    """

    # Remove singleton dimension and mask pixels with NaN values
    data = np.ma.array(data.squeeze(), mask=np.isnan(data))
    rval, cval = data.shape

    # Retrieve user-defined keyword arguments
    xaxis = kwds.pop('x', np.arange(0, rval))
    yaxis = kwds.pop('y', np.arange(0, cval))
    figuresize = kwds.pop('figsize', 1.)
    # default colormap is reverse terrain
    cmap = kwds.pop('colormap', 'terrain_r')
    xlabel = kwds.pop('xlabel', '')
    ylabel = kwds.pop('ylabel', '')
    axislabelsize = kwds.pop('ax_labelsize', 12)
    ticklabelsize = kwds.pop('tk_labelsize', 10)

    # Generate plot frame
    try:
        fw, fh = numFormatConversion(figuresize)
    except:
        fw, fh = 2 * figaspect(data)
    
    figure, ax = plt.subplots(1, figsize=(fw, fh))

    # Use pcolormesh to render 2D plot
    ygrid, xgrid = np.meshgrid(yaxis, xaxis)
    ax.pcolormesh(xgrid, ygrid, data, cmap=cmap)

    # Set basic axis properties
    ax.set_xlabel(xlabel, fontsize=axislabelsize)
    ax.set_ylabel(ylabel, fontsize=axislabelsize)
    ax.tick_params(labelsize=ticklabelsize)
    plt.tight_layout()

    return ax


def ysplitplot(datamat, xaxis, yaxis, ysplit=160):
    """
    Split-screen plot of an ARPES spectrum (intensity scaled differently for valence and conduction bands)

    **Parameters**

    datamat : numeric 2D array
            the 2D data matrix to plot
    xaxis : numeric 1D array
            the x axis coordinates
    yaxis : numeric 1D array
            the y axis coordinates
    ysplit : int
            the index of the split y position

    **Returns**
    
    axu : axes object
        handle for the upper subplot axes
    axl : axes object
        handle for the lower subplot axes
    """

    r, c = datamat.shape

    datalow = datamat[:ysplit, :]
    datahi = datamat[ysplit:, :]
    hdiv = int(np.round(60 * (1 - ysplit / r)))

    gs = matgrid.GridSpec(60, 40)

    # Calculate energy scale
    Emin, Emax = np.min(yaxis), np.max(yaxis)
    Estep = (Emax - Emin) / r
    Elow = Emin + Estep * np.arange(ysplit)
    Ehi = np.max(Elow) + Estep * np.arange(r - ysplit)

    # Calculate the angular scale
    Angmin, Angmax = np.min(xaxis), np.max(xaxis)
    Angstep = (Angmax - Angmin) / c
    Angaxis = Angstep * (np.arange(c) - c / 2)

    fig = plt.figure(num=None, figsize=(8, 8))

    # Plot the upper split graph (axu = axis upper)
    axu = plt.subplot(gs[0:hdiv - 1, :37])
    axcbh = plt.subplot(gs[0:hdiv - 1, 38:40])
    x, y = np.meshgrid(Angaxis, Ehi)
    arpesmin, arpesmax = np.min(datahi[:]), 1.2 * np.max(datahi[:])
    levscale = np.linspace(arpesmin, arpesmax, 20, endpoint=True)
    cnth = axu.contourf(
        x,
        y,
        datahi,
        levels=levscale,
        cmap='Blues',
        vmin=arpesmin,
        vmax=arpesmax *
        0.75,
        zorder=1)
    axu.contour(
        x,
        y,
        datahi,
        levscale,
        colors='k',
        linewidths=np.linspace(
            0.3,
            1.2,
            20))
    axu.set_xticks([])
    axu.set_ylabel('Energy (eV)', fontsize=20)
    cbarh = plt.colorbar(cnth, cax=axcbh)
    cbarh.set_ticks([])

    # Plot the lower split graph (axl = axis lower)
    axl = plt.subplot(gs[hdiv:60, :37])
    axcbl = plt.subplot(gs[hdiv:60, 38:40])
    x, y = np.meshgrid(Angaxis, Elow)
    arpesmin, arpesmax = np.min(datalow[:]), 1.3 * np.max(datalow[:])
    levscale = np.linspace(arpesmin, arpesmax, 30, endpoint=True)
    cntl = axl.contourf(
        x,
        y,
        datalow,
        levels=levscale,
        cmap='Blues_r',
        vmin=arpesmin,
        vmax=arpesmax * 0.9,
        zorder=1)
    axl.contour(
        x,
        y,
        datalow,
        levscale,
        colors='k',
        linewidths=np.linspace(
            1.2,
            0.3,
            25))
    axl.set_xlabel(r'Angle ($^{\circ}$)', fontsize=20)
    cbarl = plt.colorbar(cntl, cax=axcbl)
    cbarl.set_ticks([])

    fig.subplots_adjust(hspace=0)

    return [axu, axl]


def sliceview3d(datamat, axis=0, numbered=True, **kwds):
    """
    3D matrix slices displayed in a grid of subplots
    **Parameters**
    
    datamat : numeric 3D array
            the 3D data to plot
    axis : int
        the axis to slice through
    numbered : bool
        condition for numbering the subplots
    **kwds : keyword arguments
        ==========  ==========  =====================================
        keyword     data type   meaning
        ==========  ==========  =====================================
        ncol        int         number of columns in subplot grid
        nrow        int         number of rows in subplot grid
        figsize     tuple/list  figure size, (vertical_size, horizontal_size)
        colormap    str         `matplotlib colormap string <https://matplotlib.org/users/colormaps.html>`_ 
        cscale      str         colormap scaling ('log', 'linear', or 'gammaA-b', see below)
        numcolor    str         color code for subplot number
        numsize     int         fontsize of the subtitle within a subplot
        wspace      float       width spacing between subplots
        hspace      float       height spacing betweens subplots
        maintitle   str         main title of the plot
        axisreturn  str         'flattened' or 'nested', return format of axis object
        ==========  ==========  =====================================
    **Return**
    
    ax : axes object
        handle for the subplot axes
    """
    
    # Mask pixels with NaN values
    datamat = np.ma.array(datamat.squeeze(), mask=np.isnan(datamat))
    
    # Gather parameters from input
    cutdim = datamat.shape[axis]
    nc = kwds.pop('ncol', 4)
    nr = kwds.pop('nrow', np.ceil(cutdim / nc).astype('int'))
    cmap = kwds.pop('colormap', 'Greys')
    cscale = kwds.pop('cscale', 'log')
    numcolor = kwds.pop('numcolor', 'black')
    figuresize = kwds.pop('figsize', '')
    numsize = kwds.pop('numsize', 15)
    ngrid = nr * nc

    # Construct a grid of subplots
    try:
        fw, fh = numFormatConversion(figuresize)
    except:
        fw, fh = 5 * figaspect(np.zeros((nr, nc)))

    f, ax = plt.subplots(nrows=nr, ncols=nc, figsize=(fw, fh))

    # Put each figure in a subplot, remove empty subplots
    for i in range(ngrid):

        # Select the current axis based on the index
        axcurr = ax[np.unravel_index(i, (nr, nc))]

        if i <= cutdim - 1:
            # Roll the slicing axis to the start of the matrix before slicing
            img = np.rollaxis(datamat, axis)[i, :, :]

            # Set color scaling for each image individually
            if cscale == 'log':  # log scale
                im = axcurr.imshow(img, cmap=cmap)
                im.set_norm(mpl.colors.LogNorm())
            elif cscale == 'linear':  # linear scale
                im = axcurr.imshow(img, cmap=cmap)
                im.set_norm(mpl.colors.Normalize())
            elif 'gamma' in cscale:  # gamma scale
                gfactors = re.split('gamma|-', cscale)[1:]
                gfactors = numFormatConversion(gfactors, form='float')
                img = gfactors[0]*(img**gfactors[1])
                im = axcurr.imshow(img, cmap=cmap)
            
            # to do: set global color scaling for 3D matrix

            axcurr.get_xaxis().set_visible(False)
            axcurr.get_yaxis().set_visible(False)

            if numbered:
                axcurr.text(
                    0.03,
                    0.92,
                    '#%s' %
                    i,
                    fontsize=numsize,
                    color=numcolor,
                    transform=axcurr.transAxes)
        else:
            f.delaxes(axcurr)

    # Add the main title
    figtitle = kwds.pop('maintitle', '')
    if figtitle:
        f.text(
            0.5,
            0.955,
            figtitle,
            horizontalalignment='center',
            fontproperties=FontProperties(
                size=20))

    wsp = kwds.pop('wspace', 0.05)
    hsp = kwds.pop('hspace', 0.05)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=wsp,
        hspace=hsp)
    
    axisreturn = kwds.pop('axisreturn', 'flattened')
    if axisreturn == 'nested':
        return ax
    elif axisreturn == 'flattened':
        return ax.ravel()


#==========#
# 3D plots #
#==========#

def surf2d(datamat, **kwds):
    """
    2D surface plot
    """
    
    rval, cval = datamat.shape
    y, x = np.meshgrid(np.arange(cval), np.arange(rval))
    mlab.figure(bgcolor=(1,1,1))
    s = mlab.surf(x, y, datamat/np.max(datamat), warp_scale='auto')
    
    return s