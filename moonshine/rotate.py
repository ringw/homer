import numpy as np
from .opencl import *

HOUGH_NUM_THETA = 11 # should be odd
NUM_WORKERS = 32
def hough_score(page, thetas):
    """ Find the angle with the maximum sum of squares of each Hough bin.
        This should correspond to the strongest horizontal lines.
    """
    bins = hough_line(page.img, rhores=1, numrho=page.img.shape[0], thetas=thetas)
    b = bins.get()
    from pylab import *
    imshow(b)
    colorbar()
    show()
    return (bins.astype(np.float32) ** 2).get().sum(1)

def rotate(page):
    ts = np.linspace(-np.pi/100, np.pi/100, HOUGH_NUM_THETA)
    return hough_score(page, ts)
