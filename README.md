# mpes

![Build Status](https://www.travis-ci.org/RealPolitiX/mpes.svg?branch=master) ![License](https://img.shields.io/github/license/mpes-kit/mpes?color=lightgrey) [![Downloads](https://pepy.tech/badge/mpes)](https://pepy.tech/project/mpes)

Distributed data processing routines for multidimensional photoemission spectroscopy (MPES), an upgrade of the angle-resolved photoemission spectroscopy (ARPES) to achieve parallel data acquisition on multiple parameters by the use of a time-of-flight tube and a multichannel delay-line detector.

![Banner](https://github.com/mpes-kit/mpes/blob/master/resources/figures/Schematic.png)

In a photoemission process, an extreme UV or X-ray photon liberates an electron from the confines of the electronic potential within the material. [ARPES](https://en.wikipedia.org/wiki/Angle-resolved_photoemission_spectroscopy) directly measures the electronic energy and momentum parallel to the surface of the sample under study to infer the electronic states of the material. For a tutorial review on ARPES and its applications in physics and material science, see [here](http://www.phas.ubc.ca/~damascel/ARPES_Intro.pdf). The data structure of ARPES is a stack of 2D images measured at different sample geometries, which are used to reconstruct the full static 3D band structure of the material.


The MPES instrument enables sampling of the multidimensional parameter space associated with the electronic band structure at an elevated speed. At the minimum, it measures the two parallel momenta and the energy of photoelectrons simultaneously. The measurement procedure can be extended with recording of varying external parameters such as the temperature, photon polarization, dynamical time delay as in a time-resolved ARPES ([trARPES](http://ac.els-cdn.com/S036820481400108X/1-s2.0-S036820481400108X-main.pdf?_tid=00fe4a76-705f-11e7-aa2e-00000aacb35f&acdnat=1500894080_b61b6aadc82bb357e2797ddac6419991)) experiments using a ultrafast laser system (~ fs resolution), etc. These different flavors of momentum-resolved photoemission experiment together yield a complete understanding of the electronic properties of materials under equilibrium and nonequilibrium conditions for realistic design and simulation of electronic devices.

### Installation

1. Install from scratch

```
pip install git+https://github.com/mpes-kit/mpes.git
```
2. Upgrade or overwrite an existing installation

```
pip install --upgrade git+https://github.com/mpes-kit/mpes.git
```

3. PyPI installation

```
pip install mpes
```

4. Install a specific version

```
# version 1.0.9 from PyPI
pip install mpes==1.0.9

# version 0.9.8 from GitHub
pip install --upgrade git+https://github.com/mpes-kit/mpes.git@0.9.8
```

### Documentation & citation

Documentation on the usage is posted [here](https://mpes-kit.github.io/mpes/) and examples are provided in [Jupyter notebooks](https://github.com/mpes-kit/mpes/tree/master/examples).

If you use it in your work, please cite the latest version of the paper [arXiv:1909.07714](https://arxiv.org/abs/1909.07714).

The size of the single-event data is in the GB to TB range, some datasets used for the example [notebooks](https://github.com/mpes-kit/mpes/tree/master/examples) are made available on a [Zenodo repository](https://doi.org/10.5281/zenodo.3987303). Please always use the latest version of the datasets.