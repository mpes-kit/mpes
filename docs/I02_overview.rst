Overview of modules
=======================


The mpes package contains the following major modules. They are listed here along with suggested import conventions


.. code-block:: python

    import mpes.fprocessing as fp  
    import mpes.analysis as aly
    import mpes.visualization as vis


which can also be done without using the aliases simply as (not recommended):


.. code-block:: python

    from mpes import *


Description
##############

``mpes.base``: Contains the base classes that the data processing classes build on.

``mpes.fprocessing``: Contains functions and classes for data format conversion and distributed single-event data processing.

``mpes.analysis``: Contains components for data calibration, data slicing (e.g. along momentum path, or volumetric cut-out), artifact correction, peak detection, lineshape and background models for elementary data fitting and simulation.

``mpes.bandstructure``: Contains data structure classes (subclasses of ``xarray.DataArray``) that support multidimensional data manipulation befitting photoemission spectroscopy.

``mpes.visualization``: Contains functions for visualization (static and dynamic or movie).

``mpes.utils`` and ``mpes.mirrorutil``: Contain utility functions supporting various other modules.