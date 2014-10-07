from .base import BaseStaves
from ..gpu import *
from ..page import Page
from .. import util

class DummyStaves(BaseStaves):
    """ Detects and removes staves using a labeled image with the staff removed.
    """
    page = None
    nostaff_img = None
    staffonly_img = None
    def __init__(self, page, nostaff):
        """ nostaff should probably be a Page instance """
        self.page = page
        if isinstance(nostaff, Page):
            self.nostaff_img = nostaff.img
        else:
            self.nostaff_img = nostaff
        self.staffonly_img = self.nostaff_img.get() ^ self.page.bitimg

    def get_staves(self):
        any_pixel = self.staffonly_img.any(axis=1)
        is_seg_start = any_pixel.copy()
        is_seg_start[1:] &= ~ any_pixel[:-1]
        is_seg_end = any_pixel.copy()
        is_seg_end[:-1] &= ~ any_pixel[1:]
        seg_start, = np.where(is_seg_start)
        seg_end, = np.where(is_seg_end)
        if len(seg_start) == 0:
            self.staves = np.ma.zeros((0, 2, 2))
            return self.staves
        seg_gap = np.zeros_like(seg_start, bool)
        seg_gap[1:] = (seg_start[1:] - seg_end[:-1]) >= self.page.staff_dist*2
        seg_id = np.cumsum(seg_gap)
        staves = np.zeros((seg_id.max() + 1, 2, 2), np.int32)
        for i in xrange(len(staves)):
            which_runs, = np.where(seg_id == i)
            ymin = seg_start[which_runs[0]]
            ymax = seg_end[which_runs[-1]]
            ycenter = np.mean([ymin, ymax])
            staves[i] = [[0, ycenter], [self.page.orig_size[1], ycenter]]
        self.staves = np.ma.array(staves)
        return self.staves
