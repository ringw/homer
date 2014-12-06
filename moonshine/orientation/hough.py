""" Determine rotation of page using fuzzy staff detection.
    This can miss some of the staves, but it must have a high tolerance
    for rotation. Therefore, we can use the Hough staff detector with
    user-defined tolerance and resolution parameters.
"""
from ..gpu import *
from .. import bitimage, hough
from ..staves.hough import FilteredHoughStaves
from . import base
import numpy as np
from reikna.core import Type
import reikna.fft

class HoughStavesRotation(base.BaseRotation):
    staves = None
    tolerance = np.pi / 18 # 10 degrees
    resolution = np.pi / 720 # 0.25 degrees
    def get_rotation(self):
        if self.staves is None:
            self.staves = FilteredHoughStaves(self.page)
            self.staves.thetas = np.linspace(-self.tolerance, +self.tolerance,
                                             2*int(self.tolerance
                                                       / self.resolution) + 1)
        lines = self.staves.get_hough_peak_lines()
        angles = [np.arctan2(y1 - y0, x1 - x0)
                  for ((x0, y0), (x1, y1)) in lines]
        rotation = -np.median(angles)
        if np.isnan(rotation):
            return 0.0
        else:
            return rotation
