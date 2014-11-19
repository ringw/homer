from .base import BaseStaves
from ..gpu import *
from ..page import Page
from .. import util

import json
import logging

class LabeledStaffPosition(BaseStaves):
    """ Detects and removes staves using a json file with the positions of
        each staff. """
    page = None
    staff_dist = None
    staves_list = None
    def __init__(self, page, staff_pos):
        self.page = page
        if type(staff_pos) is not list:
            staff_pos = json.load(staff_pos)
        self.staves_list = staff_pos[0]

    def get_staves(self):
        self.staff_dist = np.median([s['staffspace'] for s in self.staves_list])
        if (abs(self.staff_dist - self.page.staff_dist)
                > max(self.staff_dist, self.page.staff_dist) / 10.0):
            logging.warn('Manually labeled staff_dist seems incorrect')
        lines = [s['center'] for s in self.staves_list]
        num_segs = max([len(l) for l in lines])
        staves = np.ma.empty((len(lines), num_segs, 2), dtype=np.int32,
                             fill_value=-1)
        staves.mask = np.ones_like(staves, dtype=bool)
        for i in xrange(len(lines)):
            staves[i, :len(lines[i])] = lines[i]
            staves.mask[i, :len(lines[i])] = 0
        self.staves = staves
        return staves

class LabeledStaffRemoval(BaseStaves):
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
        seg_gap = np.zeros_like(seg_start, dtype=bool)
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
