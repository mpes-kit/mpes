How to start
============


Installation
#############


The ``mpes`` package and its main dependencies can be installed in the following ways:


Install from git repository


.. code-block:: bash

	pip install git+https://github.com/RealPolitiX/mpes.git


Install and upgrade to the latest version


.. code-block:: bash

	pip install --upgrade git+https://github.com/mpes-kit/mpes.git


Dependencies
#############


The following are the main dependencies


.. code-block:: html

	numpy = 0.13.0 +
	scipy = 0.19.0 +
	scikit-image = 0.13.0 +
	pandas = 0.19.2 +
	dask >= 0.18.0
	xarray
	opencv


Other dependencies essential for loading the submodules are described in the `requirements <https://github.com/mpes-kit/mpes/blob/master/requirements.txt>`_ file. There are situations where additional dependencies are needed to execute certain functionalities that are not essential for general use. The users need to install the dependencies separately should they wish to use them, otherwise a warning would be thrown to remind the user. This construct reduces the potential conflicts between dependencies and permits the main functionalities of the package to be run without being affected by minor dependencies.


File formats
#############

The main format of the raw data from an MPES experiment is in HDF5, which stored the single-electron events as 2D arrays (electron index vs measured axis values) and associated machine parameters. The package also supports file I/O of standard file formats (.pxp and .ibw) from `Igor Pro <https://www.wavemetrics.com/products/igorpro/igorpro.htm>`_, the "native language" of photoemission data analysis. This allows to use some of the functionalities of the ``mpes`` package for analyzing ARPES data generated from hemispherical analyzers, if needed.