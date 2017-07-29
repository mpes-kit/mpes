#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as matgrid


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

# def linplot(datamat, xstep)


#==========#
# 2D plots #
#==========#

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
        imgsize        tuple/list  (horizontal_size, vertical_size)
        x              1D array    x axis coordinates
        y              1D array    y axis coordinates
        xlabel         str         x axis label
        ylabel         str         y axis label
        cmap           str         `matplotlib colormap string <https://matplotlib.org/users/colormaps.html>`_ 
        axislabelsize  int         font size of axis text labels
        ticklabelsize  int         font size of axis tick labels
        =============  ==========  ===================================

    **Return**
    
    ax : axes object
        handle for the plot axes
    """

    # Remove singleton dimension from matrix
    data = np.squeeze(data)
    rval, cval = data.shape

    # Retrieve user-defined keyword arguments
    xaxis = kwds.pop('x', np.arange(0, rval))
    yaxis = kwds.pop('y', np.arange(0, cval))
    imgsize = kwds.pop('imgsize', 1.)
    # default colormap is reverse terrain
    cmap = kwds.pop('colormap', 'terrain_r')
    xlabel = kwds.pop('xlabel', '')
    ylabel = kwds.pop('ylabel', '')
    axislabelsize = kwds.pop('ax_labelsize', 12)
    ticklabelsize = kwds.pop('tk_labelsize', 10)

    # Generate plot frame
    if len([imgsize]) == 1:  # When specify automatic image size determination
        shape_tpl = imgsize * plt.figaspect(data)[::-1]
    elif len(imgsize) == 2:  # When specify image size manually
        shape_tpl = np.asarray(imgsize)
        print(shape_tpl)
    else:
        print('Too many input arguments!')
    figure, ax = plt.subplots(1, figsize=shape_tpl)

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
        ==========  =========  =====================================
        keyword     data type  meaning
        ==========  =========  =====================================
        ncol        int        number of columns in subplot grid
        nrow        int        number of rows in subplot grid
        cmap        str        `matplotlib colormap string <https://matplotlib.org/users/colormaps.html>`_ 
        cscale      str        colormap scaling ('log' or 'linear')
        numcolor    str        color code for subplot number    
        wspace      float      width spacing between subplots
        hspace      float      height spacing betweens subplots
        maintitle   str        main title of the plot
        ==========  =========  =====================================

    **Return**
    
    ax : axes object
        handle for the subplot axes
    """

    # Gather parameters from input
    cutdim = datamat.shape[axis]
    nc = kwds.pop('ncol', 4)
    nr = kwds.pop('nrow', np.ceil(cutdim / nc).astype('int'))
    cmap = kwds.pop('cmp', 'Greys')
    cscale = kwds.pop('cscale', 'log')
    numcolor = kwds.pop('numcolor', 'black')
    ngrid = nr * nc

    # Construct a grid of subplots
    fw, fh = 5 * figaspect(np.zeros((nr, nc)))
    f, ax = plt.subplots(nrows=nr, ncols=nc, figsize=(fw, fh))

    # Put each figure in a subplot, remove empty subplots
    for i in range(ngrid):

        # Select the current axis based on the index
        axcurr = ax[np.unravel_index(i, (nr, nc))]

        if i <= cutdim - 1:
            # Roll the slicing axis to the start of the matrix before slicing
            img = np.rollaxis(datamat, axis)[i, :, :]
            im = axcurr.imshow(img, cmap=cmap)

            # Set color scaling for each image individually
            if cscale == 'log':
                im.set_norm(mpl.colors.LogNorm())
            elif cscale == 'linear':
                im.set_norm(mpl.colors.Normalize())

            axcurr.get_xaxis().set_visible(False)
            axcurr.get_yaxis().set_visible(False)

            if numbered:
                axcurr.text(
                    0.03,
                    0.92,
                    '#%s' %
                    i,
                    fontsize=15,
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

    return ax.ravel()
