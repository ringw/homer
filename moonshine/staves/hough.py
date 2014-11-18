from ..gpu import *
from .. import filter, hough, bitimage
from ..cl_util import max_kernel
from base import BaseStaves
import numpy as np

class FilteredHoughStaves(BaseStaves):
    def get_hough_peak_lines(self):
        staff_filt = self.staff_center_filter()
        thetas = np.linspace(-np.pi/250, np.pi/250, 201)
        rhores = (self.page.staff_thick + 1) // 2
        bins = hough.hough_line_kernel(staff_filt,
                              rhores=rhores,
                              numrho=self.page.img.shape[0] // rhores,
                              thetas=thetas)
        peaks = hough.houghpeaks(bins,
                                 invalidate=(401,
                                             self.page.staff_dist*12 // rhores),
                                 thresh=bins.get().max() / 2.0)
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
        staves = self()
        assert staves is not None
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

        staff = staves[staff_num].compressed().reshape([-1, 2])
        staff[:,1] -= staff_min

        img = self.page.img.copy()
        for i in xrange(staff_max-staff_min):
            img[i, staff[0,0] // 8 : staff[-1,0] // 8] = 0
        staff_filt = self.staff_center_filter(self.page, img)

        thetas = np.linspace(-np.pi/500, np.pi/500, 51)
        rhores = 1
        bins = hough.hough_line_kernel(staff_filt,
                              rhores=rhores,
                              numrho=img.shape[0] // rhores,
                              thetas=thetas)
        peaks = hough.houghpeaks(bins,
                                 invalidate=(5, 5),
                                 thresh=bins.get().max() * 0.75)
        theta = thetas[peaks[:, 0]]
        rho = peaks[:, 1] * rhores
        x0 = 0
        x1 = self.page.orig_size[1]
        y0 = (rho*rhores) / np.cos(theta)
        y1 = (rho*rhores - x1 * np.sin(theta)) / np.cos(theta)
        for y0val, y1val in zip(y0, y1):
            # Try removing this line from the current image
            segment, _ = self.refine_and_remove_staves(
                staves=np.ma.array([[x0, y0val], [x1, y1val]],
                                   mask=False, fill_value=-1),
                img=img,
                refine_staves=True,
                remove_staves=False)
            segment = segment.compressed().reshape([-1, 2])
            if segment.shape[0] <= 1:
                continue

            # Now insert segment into the current staff, and try removing
            # the new staff line from the original image.
            # If we removed more pixels, assume it's improved over the previous
            # staff, and update the staff.
            seg0_ind, = np.where(staff[:,0] >= segment[0,0])
            if len(seg0_ind):
                seg0_ind = seg0_ind[0]
            else:
                seg0_ind = None
            seg1_ind, = np.where(staff[:,0] <= segment[-1,0])
            if len(seg1_ind):
                seg1_ind = seg1_ind[-1]
            else:
                seg1_ind = None
            if seg0_ind is None and seg1_ind is None:
                # Refuse to replace whole line as we assumed previous line
                # was best fit (it had the highest Hough peak)
                continue
            staff2 = np.concatenate(
                        ([staff[:seg0_ind]] if seg0_ind is not None else [])
                      + [segment]
                      + ([staff[seg1_ind+1:]] if seg1_ind is not None else []))
            staff2_array = np.ma.array([staff2], mask=False).astype(np.int32)

            # Now try removing the new staff. If we remove more pixels than
            # in the previous staff, assume we have a better staff.
            new_staff, new_filt = self.refine_and_remove_staves(
                staves=staff2_array, img=self.page.img[staff_min:staff_max],
                refine_staves=True, remove_staves=True)
            if not len(new_staff):
                continue

            if bitimage.as_hostimage(new_filt).sum() < bitimage.as_hostimage(img).sum():
                staff = new_staff[0].compressed().reshape([-1,2])
                img = self.page.img.copy()
                for i in xrange(staff_max-staff_min):
                    img[i, staff[0,0]//8 : staff[-1,0]//8] = 0
                staff_filt = self.staff_center_filter(self.page, img)
        staff[:,1] += staff_min
        return staff

    def extend_staves(self):
        new_staves = [self.extend_staff(i) for i in xrange(len(self.staves))]
        num_segments = max([s.shape[0] for s in new_staves])
        staves = np.ma.empty((len(new_staves), num_segments, 2),
                             dtype=np.int32,
                             fill_value=-1)
        staves.mask = np.ones_like(staves, dtype=bool)
        for i, staff in enumerate(new_staves):
            staves[i, :len(staff)] = staff
            staves.mask[i, :len(staff)] = False
        self.staves = staves

    def get_staves(self):
        self.staves = self.get_staff_segments()
        return self.staves

    def show_staff_filter(self):
        import pylab as p
        # Overlay staff line points
        staff_filt = bitimage.as_hostimage(self.staff_center_filter())
        staff_line_mask = np.ma.masked_where(staff_filt == 0, staff_filt)
        p.imshow(staff_line_mask, cmap='Greens')
