from .opencl import *
import numpy as np
import logging

def houghpeaks(H, npeaks=None, thresh=1.0, invalidate=(1, 1)):
    Hmax = maximum_filter_kernel(H).get()
    peaks = []
    for i in xrange(npeaks if npeaks is not None else 1000):
        r, t = np.unravel_index(np.argmax(Hmax), Hmax.shape)
        if Hmax[r, t] < thresh:
            break
        peaks.append((r, t))
        rmin = max(0, r - (invalidate[0] // 2))
        rmax = min(H.shape[0], r - (-invalidate[0] // 2))
        tmin = max(0, t - (invalidate[1] // 2))
        tmax = min(H.shape[1], t - (-invalidate[1] // 2))
        Hmax[rmin:rmax, tmin:tmax] = 0
    if npeaks is None:
        logging.info("houghpeaks returned %d peaks", len(peaks))
    return np.array(peaks).reshape((-1, 2))
