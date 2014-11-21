from ..gpu import *
from .. import filter, hough, bitimage
from ..cl_util import max_kernel
from base import BaseStaves
import numpy as np

class FilteredHoughStaves(BaseStaves):
    def get_hough_peak_lines(self):
        if not hasattr(self, 'thetas'):
            self.thetas = np.linspace(-np.pi/250, np.pi/250, 201)
        if not hasattr(self, 'rhores'):
            self.rhores = (self.page.staff_thick + 1) // 2
        staff_filt = self.staff_center_filter()
        bins = hough.hough_line_kernel(staff_filt,
                              rhores=self.rhores,
                              numrho=self.page.img.shape[0] // self.rhores,
                              thetas=self.thetas)
        peaks = hough.houghpeaks(bins,
                                 invalidate=(10000,
                                             self.page.staff_dist*12
                                                // self.rhores),
                                 thresh=bins.get().max() / 2.0)
        theta = self.thetas[peaks[:, 0]]
        rho = peaks[:, 1]
        x0 = 0
        x1 = self.page.orig_size[1]
        y0 = ((rho+0.5)*self.rhores) / np.cos(theta)
        y1 = ((rho+0.5)*self.rhores - x1 * np.sin(theta)) / np.cos(theta)
        left_points = np.c_[np.repeat(x0, len(y0)), y0]
        right_points = np.c_[np.repeat(x1, len(y1)), y1]
        return np.dstack([[left_points], [right_points]]).reshape((-1,2,2))

    def get_staff_segments(self):
        peaks = self.get_hough_peak_lines()
        # Refine the staves using staff_removal, but don't actually remove
        # any staves
        segments, _ = self.refine_and_remove_staves(
            staves=np.ma.array(peaks, mask=False, fill_value=-1),
            refine_staves=True,
            remove_staves=False)
        return segments
    def show_staff_segments(self):
        import pylab
        for seg in self.get_staff_segments():
            seg = seg.compressed().reshape([-1, 2])
            pylab.plot(seg[:, 0], seg[:, 1], 'y')

    def get_staves(self):
        self.staves = self.get_staff_segments()
        self.extend_staves()
        return self.staves

    def show_staff_filter(self):
        import pylab as p
        # Overlay staff line points
        staff_filt = bitimage.as_hostimage(self.staff_center_filter())
        staff_line_mask = np.ma.masked_where(staff_filt == 0, staff_filt)
        p.imshow(staff_line_mask, cmap='Greens')
