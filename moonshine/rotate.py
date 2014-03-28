import numpy as np
from .opencl import *

HOUGH_NUM_THETA = 11 # should be odd
NUM_WORKERS = 32
def hough_score(page, thetas):
    """ Find the angle with the maximum sum of squares of each Hough bin.
        This should correspond to the strongest horizontal lines.
    """
    bins = hough_line_kernel(page.img, rhores=1, numrho=page.img.shape[0], thetas=thetas)
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
    print cur_angle
