### History of major changes to mpes since the first pip-installable version

#### 2017.10.03
1. Added imrescale() to mpes.fprocessing

#### 2017.10.01
1. Added blocknorm() to mpes.analysis

#### 2017.09.20
1. Modified mpes.analysis.bootstrapfit() to include fitting in the reverse direction

#### 2017.09.17
1. Added fit_parameter_plot() to mpes.visualization
2. Added build_dynamic_matrix() to mpes.segmentation
3. Organized the code section names in major submodules (list of sections in the beginning)
4. Changed the submodule name mpes.segmentation to mpes.analysis

#### 2017.09.16
1. Added readtsv() to mpes.fprocessing
2. Added Model class to mpes.segmentation

#### 2017.09.15
1. Modified the gaussian() and voigt() functions in mpes.segmentation
2. Added func_update() and func_add() to mpes.segmentation

#### 2017.09.06
1. Added bootstrapfit() to mpes.segmentation

#### 2017.08.31
1. Now axes handle can be passed to colormesh2d()
2. Added replist, shuffleaxis to mpes.utils

#### 2017.08.29
1. Added colormap normalization

#### 2017.08.28
1. Added 'contourf' option in colormesh2d() and sliceview3d() 

#### 2017.08.18
1. Added igoribw.py to /mpes, now support igor ibw binary file

#### 2017.08.17
1. Added sortByAxes() to mpes.segmentation
2. Added revaxis() to mpes._utils

#### 2017.08.16
1. Added trisurf2d() to mpes.visualization
2. Added flipdir option to sliceview3d()

#### 2017.08.15
1. Added regionExpand() to mpes.segmentation

#### 2017.08.11
1. Added surf2d() to mpes.visualization

#### 2017.08.10
1. Added stackedlineplot() to mpes.visualization

#### 2017.08.09
1. Added shirley() to mpes.segmentation for Shirley background calculation
2. Added ridgeDetect() to mpes.segmentation for band ridge detection

#### 2017.08.08
1. Added gamma scaling to mpes.visualization.sliceview3d
2. Improved the image size auto-adjustment in mpes.visualization
3. Added numFormatConversion() to mpes.visualization

#### 2017.08.07
1. Added [`CHANGELOG.md`](https://github.com/RealPolitiX/mpes/edit/master/CHANGELOG.md).
2. Corrected the name of the colormap kwarg in mpes.visualization.sliceview3d.
3. Added file I/O support for Igor packed experiment format (pxp) using the [`igor`](https://github.com/wking/igor) package.

#### 2017.07.29
1. First full build of documentation using sphinx

#### 2017.07.27
1. First pip-installable version built using cookiecutter.
