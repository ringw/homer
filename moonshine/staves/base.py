from ..gpu import *
import numpy as np

prg = build_program("staff_removal")

class BaseStaves(object):
    page = None
    staves = None
    nostaff_img = None

    def __init__(self, page):
        self.page = page

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
    def remove_staves(self, refine=False):
        """ Default staff removal implementation, with optional refinement """
        self() # must have staves
        self.nostaff_img = self.page.img.copy()
        if refine:
            refined_num_points = np.int32(self.page.orig_size[1] / 8)
            refined_staves = thr.empty_like(Type(np.int32,
                                (self.staves.shape[0], refined_num_points, 2)))
            refined_staves.fill(-1)
        else:
            refined_num_points = np.int32(0) # disable refined_staves
            refined_staves = thr.empty_like(Type(np.int32, 1)) # dummy array
        prg.staff_removal(thr.to_device(self.staves.filled().astype(np.int32)),
                          np.int32(self.page.staff_thick+1),
                          np.int32(self.page.staff_dist),
                          self.nostaff_img,
                          np.int32(self.nostaff_img.shape[1]),
                          np.int32(self.nostaff_img.shape[0]),
                          refined_staves,
                          refined_num_points,
                          global_size=self.staves.shape[1::-1])
        if refine:
            new_staves = refined_staves.get()
            # Must move all (-1, -1) points to end of each staff
            num_points = max([sum(staff[:, 0] >= 0) for staff in new_staves])
            staves_copy = np.empty((self.staves.shape[0], num_points, 2),
                                        np.int32)
            mask = np.ones_like(staves_copy, bool)
            for i, staff in enumerate(new_staves):
                points = staff[staff[:, 0] >= 0]
                staves_copy[i, :len(points)] = points
                mask[i, :len(points)] = 0
            self.staves = np.ma.array(staves_copy, mask=mask, fill_value=-1)

    def show(self):
        import pylab as p
        for staff in self():
            xs = staff[:, 0].compressed()
            ys = staff[:, 1].compressed()
            p.plot(xs, ys, 'g')
