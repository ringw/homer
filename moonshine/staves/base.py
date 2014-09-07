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
    def remove_staves(self):
        """ Default staff removal implementation """
        self() # must have staves
        self.nostaff_img = self.page.img.copy()
        prg.staff_removal(thr.to_device(self.staves.filled().astype(np.int32)),
                          np.int32(self.page.staff_thick+1),
                          np.int32(self.page.staff_dist),
                          self.nostaff_img,
                          np.int32(self.nostaff_img.shape[1]),
                          np.int32(self.nostaff_img.shape[0]),
                          thr.empty_like(Type(np.int32, 1)), # dummy new staves
                          np.int32(0), # disable refined_staves
                          global_size=self.staves.shape[::-1])

    def show(self):
        import pylab as p
        for staff in self():
            xs = staff[:, 0].compressed()
            ys = staff[:, 1].compressed()
            p.plot(xs, ys, 'g')
