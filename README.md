# mpes

![Build Status](https://www.travis-ci.org/RealPolitiX/mpes.svg?branch=master) ![License](https://img.shields.io/github/license/mpes-kit/mpes?color=lightgrey)

Python-based data processing routines for multidimensional photoemission spectroscopy (MPES). An upgrade of the angle-resolved photoemission spectroscopy (ARPES) to achieve parallel data acquisition on multiple parameters by the use of a time-of-flight tube and a multichannel delay-line detector.

![Banner](https://github.com/mpes-kit/mpes/blob/master/resources/figures/Schematic.png)

In a photoemission process, an extreme UV or X-ray photon liberates an electron from the confines of the electronic potential within the material. [ARPES](https://en.wikipedia.org/wiki/Angle-resolved_photoemission_spectroscopy) directly measures the electronic energy and momentum parallel to the surface of the sample under study to infer the electronic states of the material. For a tutorial review on ARPES and its applications in physics and material science, see [here](http://www.phas.ubc.ca/~damascel/ARPES_Intro.pdf). The data structure of ARPES is a stack of 2D images measured at different sample geometries, which are used to reconstruct the full static 3D band structure of the material.


The MPES instrument enables sampling of a multidimensional parameter space at an elevated speed. [TrARPES](http://ac.els-cdn.com/S036820481400108X/1-s2.0-S036820481400108X-main.pdf?_tid=00fe4a76-705f-11e7-aa2e-00000aacb35f&acdnat=1500894080_b61b6aadc82bb357e2797ddac6419991) is an emerging technique that combines state-of-the-art ultrafast laser systems (~ fs resolution) with an existing ARPES experimental setup. TrARPES studies light-induced electronic dynamics such as phase transition, exciton dynamics, reaction kinetics, etc. It adds a time dimension, usually on the order of femtoseconds to nanoseconds, to the scope of ARPES measurements. Due to complex electronic dynamics, various coupling effects between the energy and momentum dimensions come into play in time. A complete understanding of the multidimensional time series from trARPES measurements can reveal dynamic constants crucial to the understanding of material properties and aid in simulation, design and further device applications.


The package also supports file I/O of standard file formats (.pxp and .ibw) from [Igor Pro](https://www.wavemetrics.com/products/igorpro/igorpro.htm), the "native language" of photoemission data analysis. Recently, the support for hdf5 files is added.

### Installation

1. Install from scratch

```
pip install git+https://github.com/mpes-kit/mpes.git
```
2. Upgrade an existing installation

```
pip install --upgrade git+https://github.com/mpes-kit/mpes.git
```

PyPI installation coming soon...

### Documentation & citation

Documentation and examples are posted [here](https://mpes-kit.github.io/mpes/).

If you use it in your work, please cite the latest version of the paper [arXiv:1909.07714](https://arxiv.org/abs/1909.07714).