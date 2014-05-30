import numpy as np
from .opencl import *
from . import image, preprocessing, structure, measure

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

    def show(self):
        import pylab as p
        from structure import staves, staffsystems, staffboundary
        p.figure()
        p.imshow(np.unpackbits(self.img.get()).reshape((PAGE_SIZE, PAGE_SIZE)))
        staves.show_staff_centers(self)
        staffsystems.show_barlines(self)
        staffboundary.show_boundaries(self)
