import numpy as np
from .opencl import *
from . import image, preprocessing, structure, measure, note

PAGE_SIZE = 4096

class Page:
    def __init__(self, image_data):
        img = image.image_array(image_data)
        padded_img = np.zeros((PAGE_SIZE, PAGE_SIZE), np.uint8)
        padded_img[:img.shape[0], :img.shape[1]] = img
        self.byteimg = padded_img
        self.orig_size = img.shape
        self.bitimg = np.packbits(padded_img).reshape((PAGE_SIZE, -1))
        self.img = cla.to_device(q, self.bitimg)

    def process(self):
        preprocessing.process(self)
        structure.process(self)
        measure.build_bars(self)
        self.notepitch_score = note.get_notepitch_score(self)

    def show(self):
        import pylab
        from . import bitimage
        from structure import staves, barlines, systems, staffboundary
        pylab.figure()
        pylab.imshow(bitimage.as_hostimage(self.img))
        pylab.ylim([self.orig_size[0], 0])
        pylab.xlim([0, self.orig_size[1]])
        staves.show_staff_centers(self)
        barlines.show_barlines(self)
        systems.show_system_barlines(self)
        staffboundary.show_boundaries(self)
