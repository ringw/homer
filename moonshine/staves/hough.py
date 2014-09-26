from ..gpu import *
from .. import filter, hough, bitimage
from ..cl_util import max_kernel
from base import BaseStaves
import numpy as np

class FilteredHoughStaves(BaseStaves):
    staff_filt = None
    def get_hough_peak_lines(self):
        if self.staff_filt is None:
            self.staff_filt = filter.staff_center(self.page)
        thetas = np.linspace(-np.pi/500, np.pi/500, 51)
        rhores = (self.page.staff_thick + 1) // 2
        bins = hough.hough_line_kernel(self.staff_filt,
                              rhores=rhores,
                              numrho=self.page.img.shape[0] // rhores,
                              thetas=thetas)
        peaks = hough.houghpeaks(bins,
                                 invalidate=(101,
                                             self.page.staff_dist*8 // rhores),
                                 thresh=bins.get().max() / 4.0)
        theta = thetas[peaks[:, 0]]
        rho = peaks[:, 1]
        x0 = 0
        x1 = self.page.orig_size[1]
        y0 = (rho*rhores) / np.cos(theta)
        y1 = (rho*rhores - x1 * np.sin(theta)) / np.cos(theta)
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

    def extend_staff(self, staff_num):
        # Staff must have at least staff_dist space above and below
        # to nearest staves
        staves = self.staves()
        assert staves
        staff_min = staves[staff_num,:,1].min() - 3*self.page.staff_dist
        staff_max = staves[staff_num,:,1].max() + 3*self.page.staff_dist
        if not ((staves[staff_num-1, :, 1].max()
                     if staff_num > 0 else 0) + 3*self.page.staff_dist
                < staff_min < staff_max
                < (staves[staff_num+1, :, 1].min()
                        if staff_num + 1 < len(staves)
                        else self.page.img.shape[0]) - 3*self.page.staff_dist):
            logging.warn('extend_staff failed: staves too close together')
            return staves[staff_num]

        thetas = np.linspace(-np.pi/500, np.pi/500, 51)
        rhores = 1
        bins = hough.hough_line_kernel(self.staff_filt,
                              rhores=rhores,
                              numrho=self.page.img.shape[0] // rhores,
                              thetas=thetas)
        peaks = hough.houghpeaks(bins,
                                 invalidate=(5, self.page.staff_dist // rhores),
                                 thresh=bins.get().max() / 4.0)
        theta = thetas[peaks[:, 0]]
        rho = peaks[:, 1]
        x0 = 0
        x1 = self.page.orig_size[1]
        y0 = (rho*rhores) / np.cos(theta)
        y1 = (rho*rhores - x1 * np.sin(theta)) / np.cos(theta)
        for y0val, y1val in zip(y0, y1):
            pass

        staff_slice = self.nostaff()[staff_min:staff_max].copy()
        

    def get_staves(self):
        self.staves = self.get_staff_segments()
        return self.staves

    def show_staff_filter(self):
        import pylab as p
        # Overlay staff line points
        staff_filt = np.unpackbits(self.page.staff_filt.get()).reshape((4096, -1))
        staff_line_mask = np.ma.masked_where(staff_filt == 0, staff_filt)
        p.imshow(staff_line_mask, cmap='Greens')
