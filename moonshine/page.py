import numpy as np
from .gpu import *

# Need to define this now so that orientation can use it
PAGE_SIZE = 4096
from . import image
from . import staffsize, orientation, staves
from . import barlines, systems, staffboundary, measure#, note

class Page(object):
    def __init__(self, image_data):
        self.image_data = image_data
        img = image.image_array(image_data)
        padded_img = np.zeros((PAGE_SIZE, PAGE_SIZE), np.uint8)
        padded_img[:img.shape[0], :img.shape[1]] = img
        self.byteimg = padded_img
        self.orig_size = img.shape
        self.bitimg = np.packbits(padded_img).reshape((PAGE_SIZE, -1))
        self.img = thr.to_device(self.bitimg)

        self.staves = staves.Staves(self)

    def preprocess(self):
        staffsize.staffsize(self)
        orientation.rotate(self)
        # If we rotate significantly, the vertical difference between staff
        # lines may be slightly different
        staffsize.staffsize(self)

    def structure(self):
        if not hasattr(self, 'staff_dist'):
            self.preprocess()
        self.staves()
        barlines.get_barlines(self)
        systems.build_systems(self)
        staffboundary.boundaries(self)

    def process(self):
        self.structure()
        measure.build_bars(self)
        #self.notepitch_score = note.get_notepitch_score(self)

    def show(self, show_structure=True, show_elements=False):
        import pylab
        from . import bitimage
        pylab.figure()
        pylab.imshow(bitimage.as_hostimage(self.img))
        pylab.ylim([self.orig_size[0], 0])
        pylab.xlim([0, self.orig_size[1]])
        if show_structure:
            self.staves.show()
            barlines.show_barlines(self)
            systems.show_system_barlines(self)
            staffboundary.show_boundaries(self)
        if show_elements:
            for barsystem in self.bars:
                for system_measure in barsystem:
                    for part in system_measure:
                        part.show_elements(on_page=True)
