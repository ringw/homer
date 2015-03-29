# Fujinaga's deskewing algorithm
import numpy as np
from metaomr import bitimage
def get_strip_offsets(page, strip_size=32):
    strips = get_strips(page, strip_size)
    center = strips.shape[1] / 2
    center_strip = strips[:, center]
    strip_offsets = np.zeros(strips.shape[1], int)
    for s in xrange(center + 1, strips.shape[1]):
        offset = correl_offset(center_strip, strips[:, s])
        center_strip += np.roll(strips[:, s], offset)
        strip_offsets[s] = offset
    for s in xrange(center - 1, -1, -1):
        offset = correl_offset(center_strip, strips[:, s])
        center_strip += np.roll(strips[:, s], offset)
        strip_offsets[s] = offset
    return strip_offsets
def get_column_offsets(page, strip_size=32):
    strip_offsets = get_strip_offsets(page, strip_size)
    column_offsets = np.zeros(page.orig_size[1], int)
    column_offsets[:strip_size/2] = strip_offsets[0]
    for s in xrange(len(strip_offsets) - 1):
        column_offsets[s*strip_size + strip_size/2:(s+1)*strip_size + strip_size/2] = \
            np.rint(np.linspace(strip_offsets[s], strip_offsets[s+1], strip_size)).astype(int)
    column_offsets[len(strip_offsets) * strip_size - strip_size/2:] = strip_offsets[-1]
    return column_offsets
def deskew(page):
    column_offsets = get_column_offsets(page)
    for col in xrange(page.orig_size[1]):
        page.byteimg[col] = np.roll(page.byteimg[col], column_offsets[col])
    page.img = bitimage.as_bitimage(page.byteimg)

def get_strips(page, strip_size):
    width = (page.orig_size[1] / strip_size) * strip_size
    img = page.byteimg[:, :width]
    strip_img = img.reshape((img.shape[0], img.shape[1] / strip_size, strip_size))
    return strip_img.sum(-1)

def correl_offset(strip1, strip2, max_offset=32):
    scores = np.zeros(max_offset*2 + 1)
    strip1 = strip1 - np.mean(strip1)
    strip2 = strip2 - np.mean(strip2)
    correl = np.correlate(strip1, strip2, 'same')
    # Get correlation around valid offset range
    correl = correl[len(strip1)/2 - max_offset:len(strip1)/2 + max_offset + 1]
    return correl.argmax() - max_offset
