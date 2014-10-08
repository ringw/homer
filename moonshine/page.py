import numpy as np
from .gpu import *
from . import image
from . import staffsize, orientation, staves
from . import barlines, systems, staffboundary, measure#, note

class Page(object):
    def __init__(self, image_data):
        if isinstance(image_data, np.ndarray):
            img = image_data
        else:
            self.image_data = image_data
            img = image.image_array(image_data)
        size = max(img.shape)
        assert size <= 8192
        if size <= 4096:
            size = 4096
        else:
            size = 8192
        padded_img = np.zeros((size, size), np.uint8)
        padded_img[:img.shape[0], :img.shape[1]] = img
        self.byteimg = padded_img
        self.orig_size = img.shape
        self.bitimg = np.packbits(padded_img).reshape((size, -1))
        self.img = thr.to_device(self.bitimg)
        self.size = size

        self.staves = staves.Staves(self)

    def preprocess(self):
        staffsize.staffsize(self)
        if type(self.staff_dist) is not int:
            return
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
        self.preprocess()
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
