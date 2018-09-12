#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: R. Patrick Xian
"""

import numpy as np
import cv2


def _rotate2d(image, center, angle, scale=1):
    """ 2D matrix rotation.
    """

    rotmat = cv2.getRotationMatrix2D(center, angle=angle, scale=scale)
    # Construct rotation matrix in homogeneous coordinate
    rotmat = np.concatenate((rotmat, np.array([0, 0, 1], ndmin=2)), axis=0)

    image_rot = cv2.warpPerspective(image, rotmat, image.shape)

    return image_rot, rotmat


def _rotate3d():

    pass

class EnergyCalibrator(FileCollection):
    """
    Electron binding energy calibration workflow.
    """

    def __init__(self, files=[], folder=None, file_sorting=True):

        super().__init__(folder=folder, file_sorting=file_sorting, files=files)
        self.traces = {}

    def normalize(self, axis):

        self.normspec =

    def unmap(self):
        """ Retrieve the inverted parametric map for traces.
        """

        pass

    def featureSelect(self, trace_id, region):

        self.peak = aly.peaksearch()
        pass

    def calibrate(self):
        """ Energy calibration
        """

        self.emap = calibrateE()

    def view(self):

        pass

    def saveParameters():

        pass
