#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""
# =======================
# Sections:
# 1.  Utility functions
# 2.  1D plots
# 3.  2D plots
# 4.  3D-rendered plots
# 5.  Movie generation
# =======================

from __future__ import print_function, division
from . import utils as u
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as matgrid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import matplotlib.colors as colors
from matplotlib import cm
from bokeh.layouts import gridplot
import bokeh.plotting as pbk
import bokeh.palettes as bp
from bokeh.colors import RGB
from bokeh.io import output_notebook
from copy import copy
import re, glob as g
from PIL import Image
import imageio as imio
import natsort as nts


global PLOT3D
PLOT3D = False


# =================== #
#  Utility functions  #
# =================== #

def initmpl():
    """
    Initialize mpes plot style
    """

    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{helvet}']
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['savefig.dpi'] = 300
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


class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def cm2palette(cmapName):
    """ Convert certain matplotlib colormap (cm) to bokeh palette.

    **Parameter**\n
    cmapName: str
        Name of the colormap/palette.

    **Return**\n
    palette: list
        List of colors in hex representation (a bokoeh palette).
    """

    if cmapName in bp.all_palettes.keys():
        palette = eval('bp.' + cmapName)

    else:
        mpl_cm_rgb = (255 * eval('cm.' + cmapName)(range(256))).astype('int')
        palette = [RGB(*tuple(rgb)).to_hex() for rgb in mpl_cm_rgb]

    return palette


# ========== #
#  1D plots  #
# ========== #

def stackedlineplot(datamat, axis=0, interval=0, binning=1, **kwds):
    """
    Stacked line plots (used for visualizing energy or momentum dispersion curves)

    **Parameters**

    data: numeric 2D array
        the 2D data to plot
    axis: int | 0
        the axis to cut along
    interval: float | 0
        the interval between plots
    binning: int | 1 (no binning)
        number of binned rows/columns
    **kwds: keyword arguments
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

    ax: axes object
        handle for the plot axes
    """

    # Check binning input
    binning = int(round(binning))
    if binning < 1:
        binning = 1

    # Determine figure size
    figuresize = kwds.pop('figsize', '')
    try:
        fw, fh = u.numFormatConversion(figuresize)
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


def plot_single_hist(histvals, edges, legend=None, **kwds):
    """ Bokeh-based plotting of a single histogram with legend and tooltips.

    **Parameters**\n
    histvals: 1D array
        Histogram counts (e.g. vertical axis).
    edges: 1D array
        Histogram edge values (e.g. horizontal axis).
    legend: str
        Text for the plot legend.
    **kwds:
        Keyword arguments for 'bokeh.plotting.figure().quad()'.

    **Return**\n
    p: object
        An instance of 'bokeh.plotting.figure()' as a plot handle.
    """

    ttp = kwds.pop('tooltip', [('(x, y)', '($x, $y)')])

    p = pbk.figure(background_fill_color='white', tooltips=ttp)
    p.quad(top=histvals, bottom=0, left=edges[:-1], right=edges[1:],
           line_color='white', alpha=0.8, legend=legend, **kwds)

    p.y_range.start = 0
    p.legend.location = 'top_right'
    p.grid.grid_line_color = 'lightgrey'

    return p


def grid_histogram(dct, ncol, rvs=['X', 'Y', 't', 'ADC'], rvbins=[80, 80, 80, 80],
          rvranges=[(0, 1800), (0, 1800), (68000, 74000), (0, 500)], backend='matplotlib',
          legend=True, histkwds={}, legkwds={}, **kwds):
    """
    Grid plot of multiple 1D histograms.

    **Parameters**\n
    dct: dict
        Dictionary containing the name and values of the random variables.
    ncol: int
        Number of columns in the plot grid.
    rvs: list/tuple
        List of names for the random variables (rvs).
    rvbins: list/tuple
        Bin values for all random variables.
    rvranges: list/tuple
        Value ranges of all random variables.
    backend: str | 'matplotlib'
        Backend for making the plot ('matplotlib' or 'bokeh').
    legend: bool | True
        Option to include a legend in each histogram plot.
    histkwds: dict | {}
        Keyword arguments for histogram plots.
    legkwds: dict | {}
        Keyword arguments for legends.
    **kwds: keyword arguments
    """

    figsz = kwds.pop('figsize', (14, 8))

    if backend == 'matplotlib':

        nrv = len(rvs)
        nrow = int(np.ceil(nrv / ncol))
        histtype = kwds.pop('histtype', 'step')

        f, ax = plt.subplots(nrow, ncol, figsize=figsz)
        otherax = ax.copy()
        for i, zipped in enumerate(zip(rvs, rvbins, rvranges)):

            # Make each histogram plot
            rvname, rvbin, rvrg = zipped
            try:
                axind = np.unravel_index(i, (nrow, ncol))
                ax[axind].hist(dct[rvname], bins=rvbin, range=rvrg, label=rvname, histtype=histtype, **histkwds)
                if legend == True:
                    ax[axind].legend(fontsize=15, **legkwds)

                otherax[axind] = None

            except:
                ax[i].hist(dct[rvname], bins=rvbin, range=rvrg, label=rvname, histtype=histtype, **histkwds)
                if legend == True:
                    ax[i].legend(fontsize=15, **legkwds)

                otherax[i] = None

        for oax in otherax.flatten():
            if oax is not None:
                f.delaxes(oax)

    elif backend == 'bokeh':

        output_notebook(hide_banner=True)

        plots = []
        for i, zipped in enumerate(zip(rvs, rvbins, rvranges)):

            rvname, rvbin, rvrg = zipped
            histvals, edges = np.histogram(dct[rvname], bins=rvbin, range=rvrg)

            if legend == True:
                plots.append(plot_single_hist(histvals, edges, legend=rvname, **histkwds))
            else:
                plots.append(plot_single_hist(histvals, edges, legend=None, **histkwds))

        # Make grid plot
        pbk.show(gridplot(plots, ncols=ncol, plot_width=figsz[0]*30, plot_height=figsz[1]*28))


# ========== #
#  2D plots  #
# ========== #

def _imshow(img, plotaxes=None, xtk=None, ytk=None, xtklb=None, ytklb=None, fontsize=15, **kwds):
    """
    Generic image plotter with specifications of axes, ticks, and ticklabels.
    """

    if plotaxes is None:
        f, ax = plt.subplots()
    else:
        ax = plotaxes

    ax.imshow(img, **kwds)

    try:
        ax.set_xticks(xtk)
        ax.set_xticklabels(xtklb, fontsize=fontsize)
    except:
        pass

    try:
        ax.set_yticks(ytk)
        ax.set_yticklabels(ytklb, fontsize=fontsize)
    except:
        pass

    return f, ax


def colormesh2d(data, **kwds):
    """
    Efficient one-line color mesh plot of a 2D data matrix

    **Parameters**

    data: numeric 2D array
        The 2D data to plot.
    **kwds: keyword arguments
        =============  ==========  ===================================
        keyword        data type   meaning
        =============  ==========  ===================================
        figsize        tuple/list  (horizontal_size, vertical_size)
        x              1D array    x axis coordinates
        y              1D array    y axis coordinates
        xlabel         str         x axis label
        ylabel         str         y axis label
        colormap       str         `matplotlib colormap string <https://matplotlib.org/users/colormaps.html>`_
        cscale         str/dict    '' (default), 'linear', 'log', 'gammaA-b', dictionary with keys
                                   ['midpoint', 'vmin', 'vmax'] for bilinear normalization
        levels         1D array    explicit contour levels (ignores ncontour if not None)
        ncontour       int         number of contours (ignores levels if not None)
        plottype       str         'pcolormesh' or 'contourf'
        plotaxes       AxesObject  supply an existing AxesObject to plot on
        vmin           float       minimum value of colormap
        vmax           float       maximum value of colormap
        ax_labelsize   int         font size of axis text labels
        tk_labelsize   int         font size of axis tick labels
        =============  ==========  ===================================

    **Return**

    ax: axes object
        Handle for the plot axes.
    """

    # Remove singleton dimension and mask pixels with NaN values
    data = np.ma.array(data.squeeze(), mask=np.isnan(data))
    rval, cval = data.shape

    # Retrieve user-defined keyword arguments
    plottype = kwds.pop('plottype', 'pcolormesh')
    xaxis = kwds.pop('x', np.arange(0, rval))
    yaxis = kwds.pop('y', np.arange(0, cval))
    figuresize = kwds.pop('figsize', 1.)
    # default colormap is reverse terrain
    cmap = kwds.pop('colormap', 'terrain_r')
    cscale = kwds.pop('cscale', '') # Nonspecific scale doesn't interfere with intensity bounds (vmax, vmin)
    vmin = kwds.pop('vmin', None)
    vmax = kwds.pop('vmax', None)
    xlabel = kwds.pop('xlabel', '')
    ylabel = kwds.pop('ylabel', '')
    cbar = kwds.pop('cbar', False)
    axislabelsize = kwds.pop('ax_labelsize', 12)
    ticklabelsize = kwds.pop('tk_labelsize', 10)

    # Obtain plot figure size and aspect ratio
    try:
        fw, fh = u.numFormatConversion(figuresize)
    except:
        fw, fh = 2 * figaspect(data)

    # Check if there is a given axes object
    try:
        ax = kwds.pop('plotaxes')
    except:
        _, ax = plt.subplots(1, figsize=(fw, fh))

    # Use pcolormesh or contourf to render 2D plot
    xgrid, ygrid = np.meshgrid(yaxis, xaxis)

    if plottype == 'pcolormesh':
        p = ax.pcolormesh(xgrid, ygrid, data, \
    cmap=cmap, vmin=vmin, vmax=vmax, **kwds)

    elif plottype == 'contourf':
        origin = kwds.pop('origin', 'upper')
        lvls = kwds.pop('levels', None)
        nlvl = kwds.pop('ncontour', None)
        # Set the number of contours
        if lvls is None and nlvl is not None:
            p = ax.contourf(xgrid, ygrid, data, nlvl, \
    cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, **kwds)
        # Set directly the values of contour levels
        elif nlvl is None and lvls is not None:
            p = ax.contourf(xgrid, ygrid, data, levels=lvls, \
    cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, **kwds)
        else:
            p = ax.contourf(xgrid, ygrid, data, \
    cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, **kwds)

    else:
        raise Exception('No such plot type.')

    # Set color scaling for each image individually
    if isinstance(cscale, str):
        if cscale == 'log':  # log scale
            p.set_norm(mpl.colors.LogNorm())
        elif cscale == 'linear':  # linear scale (default)
            p.set_norm(mpl.colors.Normalize())
        elif 'gamma' in cscale:  # gamma scale
            gfactors = re.split('gamma|-', cscale)[1:]
            gfactors = u.numFormatConversion(gfactors, form='float', length=2)
            img = gfactors[0]*(data**gfactors[1])
            p.set_data(data)
    elif isinstance(cscale, dict):
        mp = cscale.pop('midpoint', 0.)
        cvmin = cscale.pop('vmin', np.min(data))
        cvmax = cscale.pop('vmax', np.max(data))
        p.set_norm(MidpointNormalize(vmin=cvmin, vmax=cvmax, midpoint=mp))

    # Set basic axis properties
    ax.set_xlabel(xlabel, fontsize=axislabelsize)
    ax.set_ylabel(ylabel, fontsize=axislabelsize)
    ax.tick_params(labelsize=ticklabelsize)

    if cbar:
        cb = plt.colorbar(p, ax=ax)
        cb.ax.tick_params(labelsize=ticklabelsize)

    plt.tight_layout()

    return p, ax


def fit_parameter_plot(data, ncol, axis=(0, 1), **kwds):
    """
    Plot of actual value, absolute and relative changes of the fitting parameters.

    **Parameters**

    data: 2D numeric array
        data for plotting
    ncol: int
        number of columns
    axis: tuple | (0, 1)
        axes for positioning the subplot grid
    **kwds: keyword arguments
        =============  ============  ===================================
        keyword        data type     meaning
        =============  ============  ===================================
        mainfigsize    tuple/list    (horizontal_size, vertical_size)
        cmaps          list of str   `matplotlib colormap string <https://matplotlib.org/users/colormaps.html>`_
        cscales        mixed list    specification of color scaling for each plot
        cbars          list of bool  specification of colorbars
        ncontours      list of int   number of contours for each subplot
        plottypes      list of str   plottype for each subplot
        =============  ============  ===================================

    **Returns**

    f: figure object
        figure handle
    ims: list
        list of plot objects
    axs: list
        list of axes objects
    """

    # Retrieve input parameters
    mainfigsize = kwds.pop('mainfigsize', (18,32))
    npl = np.prod([data.shape[idx] for idx in axis]) # total number of plots
    nrow = int(npl/ncol)
    cmaps = kwds.pop('cmaps', u.replist('terrain_r', ncol, nrow))
    cscales = kwds.pop('cscales', u.replist('linear', ncol, nrow))
    ptypes = kwds.pop('plottypes', u.replist('pcolormesh', ncol, nrow))
    cbars = kwds.pop('cbars', u.replist(False, ncol, nrow))
    ncontours = kwds.pop('ncontours', u.replist(20, ncol, nrow))

    rdata = u.shuffleaxis(data, axis, direction='front')

    f, axs = plt.subplots(nrow, ncol, figsize=mainfigsize)
    ims = copy(axs)

    # Make plots and apply plotting conditions to each
    for r in range(nrow):
        for c in range(ncol):
            ims[r][c], _ = colormesh2d(rdata[r,c,...], plotaxes=axs[r,c], \
                        colormap=cmaps[r][c], cscale=cscales[r][c], \
                        plottype=ptypes[r][c], cbar=cbars[r][c], \
                        ncontour=ncontours[r][c], **kwds)

    return f, ims, axs


def ysplitplot(datamat, xaxis, yaxis, ysplit=160):
    """
    Split-screen plot of an ARPES spectrum
    (intensity scaled differently for valence and conduction bands).

    **Parameters**

    datamat: numeric 2D array
            the 2D data matrix to plot
    xaxis: numeric 1D array
            the x axis coordinates
    yaxis: numeric 1D array
            the y axis coordinates
    ysplit: int
            the index of the split y position

    **Returns**

    axu: axes object
        handle for the upper subplot axes
    axl: axes object
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


def plot_overlay(imbot, imtop, origin='lower', **kwds):
    """
    Make an overlay plot of two images

    **Parameters**\n
    imbot: 2D array
        Image at the lower layer.
    imtop: 2D array
        Image at the top layer.
    origin: str | 'lower'
        Origin of the image ('lower' or 'upper'), following imshow convention.

    **Returns**\n
    f: fig
        Figure handle.
    ax: axes
        Figure axes object.
    ims: list
        List of image objects.
    """

    fsize = kwds.pop('figsize', (6, 6))
    cmapbot, cmaptop = kwds.pop('cmaps', ['terrain_r', 'Blues_r'])
    atop = kwds.pop('alphatop', 0.8)
    axisoff = kwds.pop('axoff', False)

    f, ax = plt.subplots(figsize=fsize)
    imb = ax.imshow(imbot, cmap=cmapbot, origin=origin)
    imt = ax.imshow(imtop, cmap=cmaptop, alpha=atop, origin=origin)
    ims = [imb, imt]

    if axisoff:
        plt.axis('off')

    return f, ax, ims


def bandpathplot(pathmap, symlabel, symid, energytk=None, energylabel=None, \
                 vline=True, noends=True, vlinekwds={}, **kwds):
    """
    Band path map (band dispersion in sampled k path within Brillouin zone).

    **Parameters**\n
    pathmap: 2D array
        Band path map.
    symlabel: list/tuple
        Labels of the symmetry points.
    symid: list/tuple/array
        Pixel indices of the symmetry points.
    energytk: list/tuple/array | None
        Energy axis ticks.
    energylabel: list/tuple | None
        Energy axis label.
    vline: bool | True
        Vertical annotation lines.
    noends: bool | True
        No vertical lines at the ends
    vlinekwds: dict | {}
        Style directives of the vertical annotation lines.
    **kwds: keyword arguments
        See ``mpes.visualization._imshow()``.

    **Return**\n
    ax: AxesObject
        Axes of the generated plot.
    """

    # fs = kwds.pop('fontsize', 15)

    # Make plot
    ax = _imshow(pathmap, xtk=symid, xtklb=symlabel, ytk=energytk, ytklb=energylabel, **kwds)

    # Draw vertical annotation lines
    if vline:

        if noends: # No annotation at the two ends
            symid = symid[1:-1]

        vlc = vlinekwds.pop('lc', 'r')
        vls = vlinekwds.pop('ls', '-.')
        vlw = vlinekwds.pop('lw', 1)

        for i in range(len(symid)):
            ax.axvline(x=symid[i], linestyle=vls, lw=vlw, color=vlc)

    return ax


def sliceview3d(datamat, axis=0, numbered=True, imkwds={}, **kwds):
    """
    3D matrix slices displayed in a grid of subplots.
    
    **Parameters**

    datamat: numeric 3D array
        The 3D data to plot.
    axis: int
        The axis to slice through.
    numbered: bool
        Option to print numbering onto the subplots.
    imkwds: dict
        Additional arguments for the image plotting functions (`imshow` and `contourf`).
    **kwds: keyword arguments
        ==========  ===========  ======================================================================================
        keyword     data type    meaning
        ==========  ===========  ======================================================================================
        aspect      str/numeric  aspect ratio of each subplot, from ['auto', 'equal' or scalar]
        ncol        int          number of columns in subplot grid
        nrow        int          number of rows in subplot grid
        figsize     tuple/list   figure size, (vertical_size, horizontal_size)
        flipdir     str          flip up-down or left-right of the matrix ('ud', 'lr')
        colormap    str          `matplotlib colormap string <https://matplotlib.org/users/colormaps.html>`_
        cscale      str          colormap scaling (''(default), 'log', 'linear', 'midpointx', or 'gammaA-b', see below)
        vmin        numeric      minimum of the color scale
        vmax        numeric      maximum of the color scale
        numcolor    str          color code for subplot number
        numpos      tuple        (Y, X) position of the text on the subplots
        numsize     int          fontsize of the subtitle within a subplot
        numtext     str          subtitle text within a subplot
        wspace      float        width spacing between subplots
        hspace      float        height spacing betweens subplots
        plottype    str          'imshow' (default) or 'contourf'
        maintitle   str          main title of the plot
        axisreturn  str          'flattened' or 'nested', return format of axis object
        ==========  ===========  ======================================================================================

    **Returns**

    ims: AxesImage object
        Handle for the images in each subplot.
    ax: AxesSubplot object
        Handle for the subplot axes.
    """

    # Mask pixels with NaN values
    datamat = np.ma.array(datamat.squeeze(), mask=np.isnan(datamat))

    # Gather parameters from input
    # Plot grid parameters
    cutdim = datamat.shape[axis]
    rdata = np.rollaxis(datamat, axis)
    nc = kwds.pop('ncol', 4)
    nr = kwds.pop('nrow', np.ceil(cutdim / nc).astype('int'))
    ngrid = nr * nc

    # Parameters at the subfigure level
    cmap = kwds.pop('colormap', 'Greys')
    cscale = kwds.pop('cscale', '') # Nonspecific scale doesn't interfere with intensity bounds (vmax, vmin)
    figuresize = kwds.pop('figsize', '')
    flipdir = kwds.pop('flipdir', '')
    vmin = kwds.pop('vmin', None)
    vmax = kwds.pop('vmax', None)
    asp = kwds.pop('aspect', 'auto')
    origin = kwds.pop('origin', 'lower')
    plottype = kwds.pop('plottype','imshow')
    xaxvis, yaxvis = kwds.pop('ax_visible', [False, False])

    # Text annotation on each plot, effective when numbered == True
    numcolor = kwds.pop('numcolor', 'black')
    numpos = kwds.pop('numpos', (0.03, 0.92))
    numtext = kwds.pop('numtext', ['#{}'.format(i) for i in range(cutdim)])
    numsize = kwds.pop('numsize', 15)

    # Construct a grid of subplots
    try:
        fw, fh = u.numFormatConversion(figuresize)
    except:
        fw, fh = 5 * figaspect(np.zeros((nr, nc)))

    ims = []
    f, ax = plt.subplots(nrows=nr, ncols=nc, figsize=(fw, fh))
    ax = np.atleast_2d(ax) # Axes for single-row figure
    if nc == 1: # Axes for single-column figure
        ax = ax.T

    # Construct list of plottype for all subplots
    if isinstance(plottype, list):
        try:
            plottype = sum(plottype, [])
        except:
            pass

    if isinstance(plottype, str):
        plottype = [plottype]*ngrid

    if 'contourf' in plottype:
        imr, imc = rdata[0,...].shape
        lvls = kwds.pop('levels', None)
        x = kwds.pop('x', range(imr))
        y = kwds.pop('y', range(imc))
        xgrid, ygrid = np.meshgrid(y, x)

    # Put each figure in a subplot, remove empty subplots
    for i in range(ngrid):

        # Select the current axis based on the index
        axcurr = ax[np.unravel_index(i, (nr, nc))]

        if i <= cutdim - 1:
            # Roll the slicing axis to the start of the matrix before slicing
            img = rdata[i, :, :]

            # Flip the image along an axis (if needed)
            if flipdir == 'ud':
                img = np.flipud(img)
            elif flipdir == 'lr':
                img = np.fliplr(img)

            # Make subplot
            if plottype[i] == 'imshow':
                im = axcurr.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect=asp, origin=origin, **imkwds)

                # Set color scaling for each image individually
                if cscale == 'log':  # log scale
                    im.set_norm(mpl.colors.LogNorm())
                elif cscale == 'linear':  # linear scale
                    im.set_norm(mpl.colors.Normalize())
                elif 'midpoint' in cscale:
                    mp = cscale['midpoint']
                    imin = cscale.pop('vmin', np.min(img))
                    imax = cscale.pop('vmax', np.max(img))
                    im.set_norm(MidpointNormalize(vmin=imin, vmax=imax, midpoint=float(mp)))
                elif 'gamma' in cscale:  # gamma scale
                    gfactors = re.split('gamma|-', cscale)[1:]
                    gfactors = u.numFormatConversion(gfactors, form='float', length=2)
                    img = gfactors[0]*(img**gfactors[1])
                    im.set_data(img)

            elif plottype[i] == 'contourf':
                im = axcurr.contourf(xgrid, ygrid, img, cmap=cmap,
                        levels=lvls, vmin=vmin, vmax=vmax, origin=origin, **imkwds)

            ims.append(im)

            # to do: set global color scaling for 3D matrix

            axcurr.get_xaxis().set_visible(xaxvis)
            axcurr.get_yaxis().set_visible(yaxvis)

            if numbered:
                axcurr.text(
                    numpos[0],
                    numpos[1],
                    numtext[i],
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
        return ims, ax
    elif axisreturn == 'flattened':
        return ims, ax.ravel()


# ================== #
#  3D-rendered plots #
# ================== #

def toggle3d(state=True, nb_backend=None, **kwds):
    """
    Switch on/off the mayavi backend.
    
    **Parameters**

    state: bool | True
        on/off state of the mayavi backend
    nb_backend: | None
        type of rendering engine choose from 'x3d' (interactive) and 'png' (static).
    **kwds: keyword arguments
        additional arguments to be passed into ``mayavi.mlab.init_notebook()``.
    """

    global PLOT3D
    PLOT3D = state
    if PLOT3D == True:
        global mlab
        from mayavi import mlab
        if nb_backend:
            mlab.init_notebook(nb_backend, **kwds)
    else:
        try:
            mlab = None
        except:
            pass


def surf2d(datamat, frame=True, miniaxes=False, **kwds):
    """
    Make 2D surface plot.

    **Parameters**

    datamat: numeric 2D array
        The 2D data to plot.
    frame: bool | True
        Option to control whether the frame is shown.
    **kwds: keyword arguments
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

    f: figure object
        Handle for the figure.
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
    2D triangulated surface plot rendered using ``mplot3d``.

    **Parameters**

    datamat: numeric 2d array
        2D data for plotting.

    **Returns**

    sf: Poly3DCollection object
        Handle for objects in the plot.
    ax: Axes object
        Handle for the axes of the plot.
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
    sf = ax.plot_trisurf(xgrid.flatten(), ygrid.flatten(), datamat.flatten(),
                    triangles=tri.triangles, cmap=colormap, antialiased=False)

    xlabel = kwds.pop('xlabel', '')
    ylabel = kwds.pop('ylabel', '')
    zlabel = kwds.pop('zlabel', '')
    lblpad = kwds.pop('labelpad', 15)
    ax.set_xlabel(xlabel, labelpad=lblpad)
    ax.set_ylabel(ylabel, labelpad=lblpad)
    ax.set_zlabel(zlabel, labelpad=lblpad)

    return sf, ax


# ================== #
#  Movie generation  #
# ================== #

def moviemaker(folder, files=None, imform='png', movform='avi', namestr='movie',
                file_sorting='forward', recursive=False, ret=False, **kwds):
    """ Generate a movie file from a stack of images

    **Parameters**\n
    folder: str
        Directory of the file folder.
    files: list/tuple
        Directory of the files to be used for generating a movie.
    imform: str | 'png'
        Format of the images.
    movform: str | 'avi'
        Format of the generated movie.
    namestr: str | 'movie'
        Namestring (including folder path) used to save the generated movie.
    file_sorting: str | 'forward'
        File-sorting direction.
    recursive: bool | False
        Option to generate a recursive movie.
    ret: bool | False
        Specify return.
    **kwds: keyword arguments
        Specific parameter settings for the vocoder.

    **Return**\n
    fnames: list of str
        List of (un)sorted filenames.
    """

    fps = kwds.pop('fps', 4)
    # loop = kwds.pop('loop', 1)

    if files is None:
        fnames = g.glob(folder + r'\*.'+imform) # Not sorted
        if file_sorting == 'forward':
            fnames = nts.natsorted(fnames, reverse=False)
        elif file_sorting == 'backward':
            fnames = nts.natsorted(fnames, reverse=True)
    else:
        fnames = files

    if recursive == True:
        fnames = fnames + fnames[:-1][::-1]

    images = []
    for _, fn in enumerate(fnames):

        im = Image.open(fn)
        images.append(np.asarray(im))

    imio.mimsave(namestr+'.'+movform, images, fps=fps, **kwds)

    if ret:
        return fnames
