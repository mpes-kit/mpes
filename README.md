# mpes

![Build Status](https://www.travis-ci.org/RealPolitiX/mpes.svg?branch=master)

Python-based data processing routines for multidimensional photoemission spectroscopy (MPES). An upgrade of the time- and angle-resolved photoemission spectroscopy (trARPES).


In a photoemission process, an extreme UV or X-ray photon liberates an electron from the confines of the electronic potential within the material. [ARPES](https://en.wikipedia.org/wiki/Angle-resolved_photoemission_spectroscopy) directly measures the electronic energy and momentum parallel to the surface of the sample under study to infer the electronic states of the material. For a tutorial review on ARPES and its applications in physics and material science, see [here](http://www.phas.ubc.ca/~damascel/ARPES_Intro.pdf). The data structure of ARPES is a stack of 2D images measured at different sample geometries, which are used to reconstruct the full static 3D band structure of the material.


[TrARPES](http://ac.els-cdn.com/S036820481400108X/1-s2.0-S036820481400108X-main.pdf?_tid=00fe4a76-705f-11e7-aa2e-00000aacb35f&acdnat=1500894080_b61b6aadc82bb357e2797ddac6419991) is an emerging technique that combines state-of-the-art ultrafast laser systems (~ fs resolution) with an existing ARPES experimental setup. TrARPES studies light-induced electronic dynamics such as phase transition, exciton dynamics, reaction kinetics, etc. It adds a time dimension, usually on the order of femtoseconds to nanoseconds, to the scope of ARPES measurements. Due to complex electronic dynamics, various coupling effects between the energy and momentum dimensions come into play in time. A complete understanding of the multidimensional time series from trARPES measurements can reveal dynamic constants crucial to the understanding of material properties and aid in simulation, design and further device applications.


The package supports file I/O of standard file formats (.pxp and .ibw) from [Igor Pro](https://www.wavemetrics.com/products/igorpro/igorpro.htm), the "native language" of photoemission data analysis. Recently, the support for hdf5 files is added.

### Installation

1. Install from scratch

```
pip install git+https://github.com/mpes-kit/mpes.git
```
2. Upgrade existing installation

```
pip install --upgrade git+https://github.com/mpes-kit/mpes.git
```

PyPI installation coming soon...

### Documentation & citation

Documentation and examples are posted [here](https://mpes-kit.github.io/mpes/).

If you use it in your work, please cite the paper [arXiv:1909.07714](https://arxiv.org/abs/1909.07714).

### Selected dependencies

```
numpy = 0.13.0 +
scipy = 0.19.0 +
scikit-image = 0.13.0 +
pandas = 0.19.2 +
dask >= 0.18.0
xarray
opencv
```
Others are described in the [requirements](https://github.com/mpes-kit/mpes/blob/master/requirements.txt).

### Overview of submodules  
The mpes package contains the following major submodules. They are listed here along with suggested import conventions
```
import mpes.fprocessing as fp  
import mpes.analysis as aly
import mpes.visualization as vis
```

### Using mayavi in Jupyter notebook
For 3D rendering of multidimensional dataset, the mpes package takes advantage of the recent version of mayavi (4.5.0 and above) with jupyter notebook support through [x3dom](https://www.x3dom.org/). Follow the steps below on the command line to allow mayavi output to be displayed on jupyter/ipython notebook

1. Install mayavi using the whl distribution [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#mayavi)
2. Install mayavi extension for Jupyter notebook
```
jupyter nbextension install --py mayavi --user
```
3. Enable the mayavi extension (this may give an error, but proceed nevertheless)
```
jupyter nbextension enable mayavi --user --py
```
4. When starting Jupyter notebook, set the backend to qt and increase the data I/O rate
```
jupyter notebook --NotebookApp.iopub_data_rate_limit=1e10 --gui=qt
```
5. Enable immediate display of mayavi figures explicitly in Jupyter notebook
```
from mayavi import mlab
mlab.init_notebook('x3d')  # Interactive rendering
or mlab.init_notebook('png')  # Static rendering
```
To reduce the installation requirements, mayavi is not loaded at start. Use mpes built-in functions to switch on/off mayavi display
```
import mpes.visualization as vis
vis.toggle3d(state=True, nb_backend)  # Switch on, nb_backend can be 'x3d', 'png', or blank
vis.toggle3d(state=False)  # Switch off
```
