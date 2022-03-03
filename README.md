# mpes

![Build Status](https://www.travis-ci.org/RealPolitiX/mpes.svg?branch=master) ![License](https://img.shields.io/github/license/mpes-kit/mpes?color=lightgrey) [![Downloads](https://pepy.tech/badge/mpes)](https://pepy.tech/project/mpes)

Distributed data processing routines for multidimensional photoemission spectroscopy (MPES), an upgrade of the angle-resolved photoemission spectroscopy (ARPES) to achieve parallel data acquisition on multiple parameters by the use of a time-of-flight tube and a multichannel delay-line detector.

![Banner](https://github.com/mpes-kit/mpes/blob/master/resources/figures/Schematic.png)

In a photoemission process, an extreme UV or X-ray photon liberates an electron from the confines of the electronic potential within the material. [ARPES](https://en.wikipedia.org/wiki/Angle-resolved_photoemission_spectroscopy) directly measures the electronic energy and momentum parallel to the surface of the sample under study to infer the electronic states of the material. For a tutorial review on ARPES and its applications in physics and material science, see [here](http://www.phas.ubc.ca/~damascel/ARPES_Intro.pdf). The data structure of ARPES is a stack of 2D images measured at different sample geometries, which are used to reconstruct the full static 3D band structure of the material.


The MPES instrument enables sampling of the multidimensional parameter space associated with the electronic band structure at an elevated speed. At the minimum, it measures the two parallel momenta and the energy of photoelectrons simultaneously. The measurement procedure can be extended with recording of varying external parameters such as the temperature, photon polarization, dynamical time delay as in a time-resolved ARPES ([trARPES](http://ac.els-cdn.com/S036820481400108X/1-s2.0-S036820481400108X-main.pdf?_tid=00fe4a76-705f-11e7-aa2e-00000aacb35f&acdnat=1500894080_b61b6aadc82bb357e2797ddac6419991)) experiments using a ultrafast laser system (~ fs resolution), etc. These different flavors of momentum-resolved photoemission experiment together yield a complete understanding of the electronic properties of materials under equilibrium and nonequilibrium conditions for realistic design and simulation of electronic devices.

### Installation

1. Install from scratch

    <pre><code class="console"> pip install git+https://github.com/mpes-kit/mpes.git
    </code></pre>

2. Upgrade or overwrite an existing installation

    <pre><code class="console"> pip install --upgrade git+https://github.com/mpes-kit/mpes.git
    </code></pre>

3. [PyPI](https://pypi.org/project/mpes/) installation

    <pre><code class="console"> pip install mpes
    </code></pre>

4. Install a specific version

    <pre><code class="console"> # version 1.0.9 from PyPI
    pip install mpes==1.0.9

    # version 0.9.8 from GitHub
    pip install --upgrade git+https://github.com/mpes-kit/mpes.git@0.9.8
    </code></pre>

### Documentation and tutorials

Documentation on the usage is posted [here](https://mpes-kit.github.io/mpes/) and examples are provided in [Jupyter notebooks](https://github.com/mpes-kit/mpes/tree/master/examples).

List of current tutorials are viewable using [nbviewer](https://nbviewer.jupyter.org) via the links

- [**Tutorial_01_HDF5 File Management**](https://nbviewer.jupyter.org/github/mpes-kit/mpes/blob/master/examples/Tutorial_01_HDF5%20File%20Management.ipynb)
- [**Tutorial_02_Data Binning**](https://nbviewer.jupyter.org/github/mpes-kit/mpes/blob/master/examples/Tutorial_02_Data%20Binning.ipynb)
- [**Tutorial_03_Rebinning Artefacts**](https://nbviewer.jupyter.org/github/mpes-kit/mpes/blob/master/examples/Tutorial_03_Rebinning%20Artefacts.ipynb)
- [**Tutorial_04_Distortion Correction**](https://nbviewer.jupyter.org/github/mpes-kit/mpes/blob/master/examples/Tutorial_04_Distortion%20Correction.ipynb)
- [**Tutorial_05_Axes Calibration**](https://nbviewer.jupyter.org/github/mpes-kit/mpes/blob/master/examples/Tutorial_05_Axes%20Calibration.ipynb)
- [**Tutorial_06_MPES_Workflow**](https://nbviewer.jupyter.org/github/mpes-kit/mpes/blob/master/examples/Tutorial_06_MPES_Workflow.ipynb)

The size of the single-event datasets used in the tutorial [notebooks](https://github.com/mpes-kit/mpes/tree/master/examples) are in the GB to TB range each, which reflect the actual examperimental setting and the light source configuration (see [here](https://doi.org/10.1063/5.0024493) for technical details). Example datasets are made available publicly in a [Zenodo repository](https://doi.org/10.5281/zenodo.3987303). Please always use the latest version of the datasets.

### Reference

If you want to refer the software in your work, please cite the following paper.

R. P. Xian, Y. Acremann, S. Y. Agustsson, M. Dendzik, K. Bühlmann, D. Curcio, D. Kutnyakhov, F. Pressacco, M. Heber, S. Dong, T. Pincelli, J. Demsar, W. Wurth, P. Hofmann, M.Wolf, M. Scheidgen, L. Rettig, R. Ernstorfer, An open-source, end-to-end workflow for multidimensional photoemission spectroscopy, [Sci. Data 7, 442 (2020)](https://www.nature.com/articles/s41597-020-00769-8).

Specifically, for the symmetry distortion correction, please cite

R. P. Xian, L. Rettig, R. Ernstorfer, Symmetry-guided nonrigid registration: The case for distortion correction in multidimensional photoemission spectroscopy, [Ultramicroscopy 202, 133 (2019)](https://doi.org/10.1016/j.ultramic.2019.04.004).
