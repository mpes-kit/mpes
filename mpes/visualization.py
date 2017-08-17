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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from mayavi import mlab
from ._utils import numFormatConversion
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
    binning = int(round(binning))
    if binning < 1:
        binning = 1
    
    # Determine figure size
    figuresize = kwds.pop('figsize', '')
    try:
        fw, fh = numFormatConversion(figuresize)
    except:
        fw, fh = 2 * figaspect(datamat)
    f, ax = plt.subplots(figsize=(fw, fh))
    
    datamat = np.rollaxis(np.asarray(datamat), axis)
    nr, nc = map(int, datamat.shape)
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
        vmin           float       minimum value of colormap
        vmax           float       maximum value of colormap
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
    vmin = kwds.pop('vmin', None)
    vmax = kwds.pop('vmax', None)
    xlabel = kwds.pop('xlabel', '')
    ylabel = kwds.pop('ylabel', '')
    axislabelsize = kwds.pop('ax_labelsize', 12)
    ticklabelsize = kwds.pop('tk_labelsize', 10)

    # Generate plot frame
    try:
        fh, fw = numFormatConversion(figuresize)
    except:
        fw, fh = 2 * figaspect(data)
    
    figure, ax = plt.subplots(1, figsize=(fw, fh))

    # Use pcolormesh to render 2D plot
    ygrid, xgrid = np.meshgrid(yaxis, xaxis)
    ax.pcolormesh(xgrid, ygrid, data, cmap=cmap, vmin=vmin, vmax=vmax)

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
        flipdir     str         flip up-down or left-right of the matrix ('ud', 'lr')
        colormap    str         `matplotlib colormap string <https://matplotlib.org/users/colormaps.html>`_ 
        cscale      str         colormap scaling ('log', 'linear', or 'gammaA-b', see below)
        vmin        numeric     minimum of the color scale
        vmax        numeric     maximum of the color scale
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
    flipdir = kwds.pop('flipdir', '')
    numsize = kwds.pop('numsize', 15)
    vmin = kwds.pop('vmin', None)
    vmax = kwds.pop('vmax', None)
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
            
            # Flip the image along an axis (if needed)
            if flipdir == 'ud':
                img = np.flipud(img)
            elif flipdir == 'lr':
                img = np.fliplr(img)
            
            # Set color scaling for each image individually
            if cscale == 'log':  # log scale
                im = axcurr.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                im.set_norm(mpl.colors.LogNorm())
            elif cscale == 'linear':  # linear scale
                im = axcurr.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                im.set_norm(mpl.colors.Normalize())
            elif 'gamma' in cscale:  # gamma scale
                gfactors = re.split('gamma|-', cscale)[1:]
                gfactors = numFormatConversion(gfactors, form='float')
                img = gfactors[0]*(img**gfactors[1])
                im = axcurr.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # to do: set global color scaling for 3D matrix

            axcurr.get_xaxis().set_visible(False)
            axcurr.get_yaxis().set_visible(False)

            if numbered:
                axcurr.text(
                    0.03,
                    0.92,
                    '#%d' %
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


#======================#
# Plots rendered in 3D #
#======================#

def surf2d(datamat, frame=True, miniaxes=False, **kwds):
    """
    2D surface plot
        **Parameters**
    
    datamat : numeric 2D array
            the 2D data to plot
    frame : bool | True
            controls whether the frame is shown
    **kwds : keyword arguments
        ==========  ==========  =====================================
        keyword     data type   meaning
        ==========  ==========  =====================================
        alpha       float       opacity value from 0 to 1
        azimuth     float       azimuthal viewing angle (default = 0)
        cmap        str         `matplotlib colormap string <https://matplotlib.org/users/colormaps.html>`_
        elevation   float       zenith viewing angle (default = 0)
        bgc         tuple/list  background color in RGB values
        kind        str         kind of surface plot, {'points', 'wireframe', 'surface'}
        framecolor  tuple/list  color of the frame
        warp_scale  float       warp scale value from 0 to 1
        ==========  ==========  =====================================
    **Return**
    
    f : figure object
        handle for the figure
    """
    
    colormap = kwds.pop('cmap', 'rainbow')
    ws = kwds.pop('warp_scale', 'auto')
    op = kwds.pop('alpha', 1.0)
    bgc = kwds.pop('bgc', (1.,1.,1.))
    rep = kwds.pop('kind', 'surface')
    xlabel = kwds.pop('xlabel', 'x')
    ylabel = kwds.pop('ylabel', 'y')
    zlabel = kwds.pop('zlabel', 'z')
#     labelsize = kwds.pop('labelsize', 10)
    
    nr, nc = datamat.shape
    y, x = np.meshgrid(np.arange(nc), np.arange(nr))
    mlab.figure(bgcolor=bgc, fgcolor=(0.,0.,0.))
    f = mlab.surf(x, y, datamat/np.max(datamat), warp_scale=ws, 
                  colormap=colormap, opacity=op, representation=rep)
    
    az = kwds.pop('azimuth', 0)
    elev = kwds.pop('elevation', 0)
    mlab.view(azimuth=0, elevation=0, distance='auto', focalpoint='auto')
    
    # Display the miniature orientation axes
    if miniaxes == True:
        mlab.orientation_axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    
    # Display the figure frame
    if frame == True:
        frc = kwds.pop('framecolor', (0,0,0))
        mlab.outline(f, color=frc, line_width=2)
#     mlab.axes(f, color=(0,0,0), xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    
    return f
    

def trisurf2d(datamat, **kwds):
    """
    2D triangulated surface plot rendered using mplot3d
    """
    
    data = np.ma.array(datamat.squeeze(), mask=np.isnan(datamat))
    rval, cval = datamat.shape
    xaxis = kwds.pop('x', np.arange(0, rval))
    yaxis = kwds.pop('y', np.arange(0, cval))
    colormap = kwds.pop('cmap', 'viridis_r')
    
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    
    ygrid, xgrid = np.meshgrid(yaxis, xaxis)
    tri = mtri.Triangulation(ygrid.flatten(), xgrid.flatten())
    ax.plot_trisurf(xgrid.flatten(), ygrid.flatten(), datamat.flatten(),
                    triangles=tri.triangles, cmap=colormap, antialiased=False)
    
    xlabel = kwds.pop('xlabel', '')
    ylabel = kwds.pop('ylabel', '')
    zlabel = kwds.pop('zlabel', '')
    lblpad = kwds.pop('labelpad', 15)
    ax.set_xlabel(xlabel, labelpad=lblpad)
    ax.set_ylabel(ylabel, labelpad=lblpad)
    ax.set_zlabel(zlabel, labelpad=lblpad)
    
    return ax