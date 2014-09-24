# Gamera interface for assessing staff detection accuracy
from . import base
import numpy as np

import gamera.core
from gamera.toolkits import musicstaves
gamera.core.init_gamera()

from PIL import Image
import sys
import tempfile

class GameraMusicStaves(base.BaseStaves):
    """ Wrapper for Gamera MusicStaves subclasses """
    gamera_class = None
    page = None
    gamera_image = None
    gamera_instance = None
    staves = None
    nostaff_img = None
    def __init__(self, page):
        self.page = page

        # Gamera must read image from a file
        gamera_img = page.byteimg[:page.orig_size[0], :page.orig_size[1]]
        gamera_img = np.where(gamera_img, 0, 255).astype(np.uint8)
        gamera_img = Image.fromarray(gamera_img).convert('1')
        tempf = tempfile.NamedTemporaryFile()
        gamera_img.save(tempf, 'png')
        tempf.flush()
        self.gamera_image = gamera.core.load_image(tempf.name)
        self.gamera_instance = self.gamera_class(self.gamera_image)

    def __call__(self):
        if self.staves is None:
            self.nostaff_img = self.gamera_instance.remove_staves(num_lines=5)
            horiz_slices = []
            # XXX: Gamera seems to return horizontal line
            staves = []
            for staff in self.gamera_instance.get_staffpos(0):
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
            mask = np.ones_like(our_staves, bool)
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
