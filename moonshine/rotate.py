import numpy as np
from .opencl import *
from . import hough

prg = build_program("rotate")
prg.rotate_image.set_scalar_arg_dtypes([
    None, # input image
    np.float32, # cos(theta)
    np.float32, # sin(theta)
    None, # output image
])

def rotate_kernel(img, theta):
    new_img = cla.zeros_like(img)
    prg.rotate_image(q, (img.shape[1], img.shape[0]),
                               (16, 8),
                               img.data,
                               np.cos(theta).astype(np.float32),
                               np.sin(theta).astype(np.float32),
                               new_img.data).wait()
    return new_img

HOUGH_NUM_THETA = 11 # should be odd
NUM_WORKERS = 32
def hough_score(page, thetas):
    """ Find the angle with the maximum sum of squares of each Hough bin.
        This should correspond to the strongest horizontal lines.
    """
    bins = hough.hough_line_kernel(page.img, rhores=1, numrho=page.img.shape[0], thetas=thetas)
    bins **= 2 # sum of squares of Hough bins
    return bins.get().sum(1)

def rotate(page):
    RANGE = np.pi/50
    cur_angle = 0.0
    for i in xrange(5):
        ts = cur_angle + np.linspace(-RANGE, RANGE, HOUGH_NUM_THETA)
        scores = hough_score(page, ts)
        cur_angle = ts[np.argmax(scores)]
        RANGE /= 5
    page.rotated = cur_angle # adjust by angle of image
    new_img = rotate_kernel(page.img, page.rotated)
    page.img = new_img
    return cur_angle
