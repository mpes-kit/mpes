.. mpes documentation master file, created by
   sphinx-quickstart on Tue Jul 25 03:22:02 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``mpes`` documentation
################################
Distributed data processing routines for multidimensional photoemission spectroscopy (MPES).


MPES concepts
================================
In a photoemission process, an extreme UV or X-ray photon liberates an electron from the confines of the electronic potential within the material. `ARPES <https://en.wikipedia.org/wiki/Angle-resolved_photoemission_spectroscopy>`_ directly measures the electronic energy and momentum parallel to the surface of the sample under study to infer the electronic states of the material. For a tutorial review on ARPES and its applications in physics and material science, see `here <http://www.phas.ubc.ca/~damascel/ARPES_Intro.pdf>`_. The data structure of ARPES is a stack of 2D images measured at different sample geometries, which are used to reconstruct the full static 3D band structure of the material.


The MPES instrument enables sampling of the multidimensional parameter space associated with the electronic band structure at an elevated speed. At the minimum, it measures the two parallel momenta and the energy of photoelectrons simultaneously. The measurement procedure can be extended with recording of varying external parameters such as the temperature, photon polarization, dynamical time delay as in a time-resolved ARPES (`trARPES <http://ac.els-cdn.com/S036820481400108X/1-s2.0-S036820481400108X-main.pdf?_tid=00fe4a76-705f-11e7-aa2e-00000aacb35f&acdnat=1500894080_b61b6aadc82bb357e2797ddac6419991>`_) experiments using a ultrafast laser system (~ fs resolution), etc. These different flavors of momentum-resolved photoemission experiment together yield a complete understanding of the electronic properties of materials under equilibrium and nonequilibrium conditions for realistic design and simulation of electronic devices.

	
.. Instructions
==================

.. toctree::
   :caption: Instructions
   :maxdepth: 1
   
   I01_start
   I02_overview
   I03_usenotes


.. API documentation
====================================

.. toctree::
   :caption: API documentation
   :maxdepth: 1

   base
   file_io
   analysis
   bandstructure
   visualization
   mirrorutil
   utils

