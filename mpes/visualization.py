#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as matgrid


def initmpl():
    """
    Initialize plot style
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


#def linplot(datamat, xstep)


def colormesh2d(data, **kwargs):
    """
    Basic color mesh plot of a 2D data matrix
    """
    
    # Remove singleton dimension from matrix
    data = np.squeeze(data)
    rval, cval = data.shape
    
    # Retrieve user-defined keyword arguments
    xaxis = kwargs.pop('x', np.arange(0, rval))
    yaxis = kwargs.pop('y', np.arange(0, cval))
    imgsize = kwargs.pop('imgsize', 1.)
    cmap = kwargs.pop('colormap', 'terrain_r') # default colormap is reverse terrain
    xlabel = kwargs.pop('xlabel', '')
    ylabel = kwargs.pop('ylabel', '')
    axislabelsize = kwargs.pop('ax_labelsize', 12)
    ticklabelsize = kwargs.pop('tk_labelsize', 10)
    
    # Generate plot frame 
    if len([imgsize]) == 1: # When specify automatic image size determination
        shape_tpl = imgsize*plt.figaspect(data)[::-1]
    elif len(imgsize) == 2: # When specify image size manually
        shape_tpl = np.asarray(imgsize)
        print shape_tpl
    else:
        print 'Too many input arguments!'
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
    Split screen plot of an ARPES spectrum
    """
    
    r, c = datamat.shape
    
    datalow = datamat[:ysplit,:]
    datahi = datamat[ysplit:,:]
    hdiv = int(np.round(60*(1-ysplit/r)))
    
    gs = matgrid.GridSpec(60, 40)
    
    # Calculate energy scale
    Emin, Emax = np.min(yaxis), np.max(yaxis)
    Estep = (Emax-Emin)/r
    Elow = Emin + Estep*np.arange(ysplit)
    Ehi = np.max(Elow) + Estep*np.arange(r-ysplit)

    # Calculate the angular scale
    Angmin, Angmax = np.min(xaxis), np.max(xaxis)
    Angstep = (Angmax - Angmin)/c
    Angaxis = Angstep*(np.arange(c) - c/2)

    fig = plt.figure(num=None, figsize=(8,8))
    
    # Plot the upper split graph (axh = axis higher)
    axh = plt.subplot(gs[0:hdiv-1, :37])
    axcbh = plt.subplot(gs[0:hdiv-1, 38:40])
    x, y = np.meshgrid(Angaxis, Ehi)
    arpesmin, arpesmax = np.min(datahi[:]), 1.2*np.max(datahi[:])
    levscale = np.linspace(arpesmin, arpesmax, 20, endpoint=True)
    cnth = axh.contourf(x, y, datahi, levels=levscale, cmap='Blues', vmin=arpesmin, vmax=arpesmax*0.75, zorder=1)
    axh.contour(x, y, datahi, levscale, colors='k', linewidths=np.linspace(0.3,1.2,20))
    axh.set_xticks([])
    axh.set_ylabel('Energy (eV)', fontsize=20)
    cbarh = plt.colorbar(cnth, cax=axcbh)
    cbarh.set_ticks([])
    
    # Plot the lower split graph (axl = axis lower)
    axl = plt.subplot(gs[hdiv:60, :37])
    axcbl = plt.subplot(gs[hdiv:60, 38:40])
    x, y = np.meshgrid(Angaxis, Elow)
    arpesmin, arpesmax = np.min(datalow[:]), 1.3*np.max(datalow[:])
    levscale = np.linspace(arpesmin, arpesmax, 30, endpoint=True)
    cntl = axl.contourf(x, y, datalow, levels=levscale, cmap='Blues_r', vmin=arpesmin, vmax=arpesmax*0.9, zorder=1)
    axl.contour(x, y, datalow, levscale, colors='k', linewidths=np.linspace(1.2,0.3,25))
    axl.set_xlabel(r'Angle ($^{\circ}$)', fontsize=20)
    cbarl = plt.colorbar(cntl, cax=axcbl)
    cbarl.set_ticks([])

    fig.subplots_adjust(hspace=0)
    
    return [axh, axl]
