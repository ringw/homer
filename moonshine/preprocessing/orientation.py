from ..opencl import *
from .. import page as page_module
PAGE_SIZE = page_module.PAGE_SIZE
import numpy as np

def patch_orientation(page, patch_size=512):
    orientations = np.zeros((PAGE_SIZE / patch_size,
                                      PAGE_SIZE / patch_size))
    for patch_y in xrange(orientations.shape[0]):
        for patch_x in xrange(orientations.shape[1]):
            patch = page.byteimg[patch_y*patch_size:(patch_y+1)*patch_size,
                                 patch_x*patch_size:(patch_x+1)*patch_size]
            patch_fft = np.abs(np.fft.fft2(patch))
            fft_top = patch_fft[:patch_size/2, :patch_size/2]
            fft_top[:200] = 0
            peak_y, peak_x = np.unravel_index(np.argmax(fft_top),
                                              fft_top.shape)
            if peak_x > patch_size/2:
                peak_x -= patch_size
            orientations[patch_y, patch_x] = np.arctan2(peak_x, peak_y)
            print peak_x, peak_y
    return orientations
