Usage notes
============


Using mayavi in Jupyter notebook
#################################

For 3D rendering of multidimensional dataset, the mpes package takes advantage of the recent version of mayavi (4.5.0 and above) with jupyter notebook support through `x3dom <https://www.x3dom.org/>`_. Follow the steps below on the command line to allow mayavi output to be displayed on jupyter/ipython notebook

#. Install mayavi using the whl distribution `here <http://www.lfd.uci.edu/~gohlke/pythonlibs/#mayavi>`_
#. Install mayavi extension for Jupyter notebook::

    jupyter nbextension install --py mayavi --user

#. Enable the mayavi extension (this may give an error, but proceed nevertheless)::

    jupyter nbextension enable mayavi --user --py

#. When starting Jupyter notebook, set the backend to qt and increase the data I/O rate::

    jupyter notebook --NotebookApp.iopub_data_rate_limit=1e10 --gui=qt

#. Enable immediate display of mayavi figures explicitly in Jupyter notebook::

    from mayavi import mlab
    mlab.init_notebook('x3d')  # Interactive rendering
    or mlab.init_notebook('png')  # Static rendering

To reduce the installation requirements, mayavi is not loaded at start. Use mpes built-in functions to switch on/off mayavi display::

    import mpes.visualization as vis
    vis.toggle3d(state=True, nb_backend)  # Switch on, nb_backend can be 'x3d', 'png', or blank
    vis.toggle3d(state=False)  # Switch off


Documentation
###################

The documentation is currently built using the ``sphinx`` package and rendered online as GitHub pages. To update the documentation, first, apply changes to existing (or create new) reStructuredText (rst) files in the ``/docs`` folder. Next, go to the ``/docs`` folder at the command line and type the following,


.. code-block:: bash

    make html


to execute the documentation building process. Most of the warnings related to line breaks may be ignored, except the fatal ones that terminates the building process. Then, check the ``/docs/index.html`` file using any standard web browser. If the changes are correctly rendered locally, update the changes to the online repository and check the outcome at the `documentation site <https://mpes-kit.github.io/mpes/>`_. Some tens of seconds of latency is expected for the changes to be included on the associated GitHub pages.