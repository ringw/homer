from ..gpu import *
import numpy as np
try:
    import scipy.signal as scipy_signal
except ImportError:
    scipy_signal = None

prg = build_program("staves")

class BaseStaves(object):
    page = None
    staves = None
    nostaff_img = None

    def __init__(self, page):
        self.page = page

    def staff_center_filter(self):
        output = thr.empty_like(self.page.img)
        prg.staff_center_filter(self.page.img,
                                np.int32(self.page.staff_thick),
                                np.int32(self.page.staff_dist),
                                output,
                                global_size=self.page.img.shape[::-1])
        return output

    def __call__(self):
        if self.staves is None:
            self.get_staves()
            if not isinstance(self.staves, np.ma.masked_array):
                self.staves = np.ma.array(self.staves, fill_value=-1)
        return self.staves

    def nostaff(self):
        if self.nostaff_img is None:
            self.remove_staves()
        return self.nostaff_img

    def get_staves(self):
        NotImplementedError("Use a concrete Staves subclass.")

    def refine_and_remove_staves(self, refine_staves=False, remove_staves=True,
                                 staves=None, img=None):
        assert refine_staves or remove_staves, 'Need something to do'
        if staves is None:
            staves = self()
        if img is None:
            img = self.page.img
        if refine_staves:
            refined_num_points = np.int32(self.page.orig_size[1] // 8)
            refined_staves = thr.empty_like(Type(np.int32,
                                (staves.shape[0], refined_num_points, 2)))
            refined_staves.fill(-1)
        else:
            refined_num_points = np.int32(0) # disable refined_staves
            refined_staves = thr.empty_like(Type(np.int32, 1)) # dummy array
        if remove_staves:
            nostaff_img = img.copy()
        else:
            nostaff_img = img
            refined_num_points = np.int32(-refined_num_points)
        prg.staff_removal(thr.to_device(staves.filled().astype(np.int32)),
                          np.int32(self.page.staff_thick+1),
                          np.int32(self.page.staff_dist),
                          nostaff_img,
                          np.int32(nostaff_img.shape[1]),
                          np.int32(nostaff_img.shape[0]),
                          refined_staves,
                          refined_num_points,
                          global_size=staves.shape[1::-1],
                          local_size=(staves.shape[1], 1))
        if refine_staves:
            if not (refined_staves != -1).any():
                return np.ma.array(np.empty([0, 2], np.int32)), nostaff_img
            new_staves = refined_staves.get()
            # Must move all (-1, -1) points to end of each staff
            num_points = max([sum(staff[:, 0] >= 0) for staff in new_staves])
            staves_copy = np.empty((staves.shape[0], num_points, 2), np.int32)
            mask = np.ones_like(staves_copy, bool)
            for i, staff in enumerate(new_staves):
                points = staff[staff[:, 0] >= 0]
                # Clean up single spurious points (requires scipy)
                if scipy_signal is not None:
                    points[:, 1] = scipy_signal.medfilt(points[:, 1])
                staves_copy[i, :len(points)] = points
                mask[i, :len(points)] = 0
            order = np.argsort(staves_copy[:, 0, 1]) # sort by y0
            staves_copy = staves_copy[order]
            mask = mask[order]
            staves = np.ma.array(staves_copy, mask=mask, fill_value=-1)
        return staves, nostaff_img

    def remove_staves(self, refine_staves=False):
        """ Default staff removal implementation, with optional refinement """
        self() # must have staves
        self.staves, self.nostaff_img = self.refine_and_remove_staves(
                remove_staves=True, refine_staves=refine_staves)

    def extract_staff(self, staff, img):
        if type(staff) is int:
            staff = self()[staff]
        if hasattr(staff, 'mask'):
            staff = staff.compressed().reshape([-1, 2])
        if img is None:
            img = self.page.img
        output = thr.empty_like(Type(np.uint8,
                    (self.page.staff_dist*4 + 1,
                     staff[-1,0]/8 + 1 - staff[0,0]/8)))
        output.fill(0)
        prg.extract_staff(thr.to_device(staff.astype(np.int32)),
                          np.int32(staff.shape[0]),
                          np.int32(self.page.staff_dist),
                          img,
                          np.int32(img.shape[1]),
                          np.int32(img.shape[0]),
                          output,
                          global_size=output.shape[::-1])
        return output

    def show(self):
        import pylab as p
        for staff in self():
            xs = staff[:, 0].compressed()
            ys = staff[:, 1].compressed()
            p.plot(xs, ys, 'g')
