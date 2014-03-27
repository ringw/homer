import numpy as np
from .opencl import *

HOUGH_NUM_THETA = 11 # should be odd
def hough_score(page, ts):
    """ Find the angle with the maximum sum of squares of each Hough bin.
        This should correspond to the strongest horizontal lines.
    """
    tan_ts = cla.to_device(q, np.tan(ts).astype(np.float32))
    numrho = page.img.shape[0]
    bins = cla.zeros(q, (HOUGH_NUM_THETA, numrho), np.uint32)
    temp = cl.LocalMemory(4 * numrho)
    hough_line.hough_line(q, (page.img.shape[1], page.img.shape[0], len(ts)),
                             (8, 1, 1),
                             page.img.data,
                             tan_ts.data,
                             np.int32(1), # rhores
                             np.int32(numrho),
                             temp,
                             bins.data).wait()
    return (bins.astype(np.float32) ** 2).get().sum(1)

def rotate(page):
    ts = np.linspace(-np.pi/100, np.pi/100, HOUGH_NUM_THETA)
    return hough_score(page, ts)
