# Gamera interface for assessing staff detection accuracy
import moonshine
from . import base
from .. import bitimage
import numpy as np

import gamera.core
import gamera.plugins.numpy_io
from gamera.toolkits import musicstaves
gamera.core.init_gamera()

from PIL import Image
import sys

class GameraMusicStaves(base.BaseStaves):
    """ Wrapper for Gamera MusicStaves subclasses """
    gamera_class = None
    page = None
    gamera_image = None
    gamera_instance = None
    staves = None
    nostaff_img = None
    gamera_staff_removal = False
    def __init__(self, page, staff_removal='moonshine'):
        self.page = page

        # Gamera must read image from a file
        gamera_img = (page.byteimg[:page.orig_size[0], :page.orig_size[1]]
                        .astype(np.uint16))
        self.gamera_image = gamera.plugins.numpy_io.from_numpy(gamera_img)
        self.gamera_instance = self.gamera_class(self.gamera_image)

        assert staff_removal in ['moonshine', 'gamera']
        if staff_removal == 'gamera':
            self.gamera_staff_removal = True

    def __call__(self):
        if self.staves is None:
            try:
                self._nostaff_img = self.gamera_instance.remove_staves(
                                         num_lines=5, undo_deskew=True)
            except TypeError: # undo_deskew only used for rl_fujinaga
                self._nostaff_img = self.gamera_instance.remove_staves(
                                         num_lines=5)
            if self._nostaff_img is None:
                # may return None, but modifies image in-place
                self._nostaff_img = self.gamera_instance.image
            horiz_slices = []
            # XXX: MusicStaves instances seem to return horizontal line
            staves = []
            staffpos = self.gamera_instance.get_staffpos(0)
            if staffpos is None:
                self.staves = np.ma.array((0, 0, 2), dtype=np.int32)
                return self.staves
            for staff in staffpos:
                ypos = staff.yposlist
                if len(ypos) != 5:
                    continue
                # Best way to convert to center line since it averages out
                # errors in each line
                ypos = np.mean(ypos)
                staves.append([[0, ypos], [self.page.orig_size[1], ypos]])
            self.staves = np.array(staves, np.int32)
            self.staves = np.ma.array(self.staves, fill_value=-1)
        return self.staves

    def refine_and_remove_staves(self, refine_staves=False, remove_staves=True,
                                 staves=None, img=None):
        if remove_staves and self.gamera_staff_removal:
            if refine_staves:
                raise NotImplementedError(
                    "Gamera staff removal with refinement not supported")
            self()
            nostaff_byte = gamera.plugins.numpy_io.to_numpy.__call__(
                    self._nostaff_img)
            nostaff_pad = np.zeros((self.page.size, self.page.size), np.uint8)
            nostaff_pad[:nostaff_byte.shape[0], :nostaff_byte.shape[1]] = \
                nostaff_byte
            self.nostaff_img = self(), bitimage.as_bitimage(nostaff_pad)
            return self.nostaff_img
        else:
            return super(GameraMusicStaves, self).refine_and_remove_staves(
                refine_staves=refine_staves, remove_staves=remove_staves,
                staves=staves, img=img)
class GameraStaffFinder(GameraMusicStaves):
    """ Wrapper for Gamera StaffFinder subclasses """

    def __call__(self):
        if self.staves is None:
            self.gamera_instance.find_staves(num_lines=5)
            if len(self.gamera_instance.linelist) == 0:
                return np.empty((0, 0, 2), np.int32)
            staves = []
            for staff in self.gamera_instance.linelist:
                staff = [line.to_skeleton() for line in staff]
                # If lines don't start at the same x, skip to the max left_x
                if np.diff([line.left_x for line in staff]).any():
                    left_x = max([line.left_x for line in staff])
                    for line in staff:
                        if line.left_x < left_x:
                            line.y_list = line.y_list[left_x - line.left_x:]
                            line.left_x = left_x
                # If lines don't end at the same x, go to the min end
                if np.diff([len(line.y_list) for line in staff]).any():
                    max_len = min([len(line.y_list) for line in staff])
                    for line in staff:
                        line.y_list = line.y_list[:max_len]
                center_y = np.mean(np.array([line.y_list for line in staff]),
                                   axis=0).astype(np.int32)
                x0 = staff[0].left_x
                our_staff = np.array([[x0 + i, y]
                                      for i, y in enumerate(center_y)
                                      if i == 0 or i+1 == len(center_y)
                                         or center_y[i-1] != y], np.int32)
                staves.append(our_staff)
            num_points = max([len(s) for s in staves])
            our_staves = np.zeros((len(staves), num_points, 2), np.int32)
            mask = np.ones_like(our_staves, dtype=bool)
            for i, staff in enumerate(staves):
                our_staves[i, :len(staff)] = staff
                mask[i, :len(staff)] = 0
            # Account for page rotation
            if hasattr(self.page, 'orientation'):
                new_staves = np.empty_like(our_staves)
                t = self.page.orientation
                new_staves[..., 0] = (our_staves[..., 0] * np.cos(t)
                                      - our_staves[..., 1] * np.sin(t))
                new_staves[..., 1] = (our_staves[..., 0] * np.sin(t)
                                      + our_staves[..., 1] * np.cos(t))
                our_staves = new_staves
            self.staves = np.ma.array(our_staves, mask=mask, fill_value=-1)
        return self.staves

MUSICSTAVES = ['MusicStaves_' + cls
               for cls in ['linetracking', 'rl_carter', 'rl_fujinaga',
                           'rl_roach_tatem', 'rl_simple', 'skeleton']]
STAFFFINDER = ['StaffFinder_' + cls
               for cls in ['dalitz', 'miyao', 'projections']]
ourmod = sys.modules[__name__]
for alg in MUSICSTAVES:
    cls = getattr(musicstaves, alg)
    ourcls = type(alg, (GameraMusicStaves,), {})
    ourcls.gamera_class = cls
    setattr(ourmod, alg, ourcls)
for alg in STAFFFINDER:
    cls = getattr(musicstaves, alg)
    ourcls = type(alg, (GameraStaffFinder,), {})
    ourcls.gamera_class = cls
    setattr(ourmod, alg, ourcls)
