from .opencl import *
import numpy as np

def houghpeaks(H, npeaks=None, threshold=None):
    Hmax = maximum_filter_kernel(H).get()
    peaks = []
    if npeaks is not None:
        for i in xrange(npeaks):
            r, t = np.unravel_index(np.argmax(Hmax), Hmax.shape)
            if Hmax[r, t] < 1:
                break
            peaks.append((r, t))
            Hmax[r, t] = 0
    elif threshold is not None:
        while True:
            r, t = np.unravel_index(np.argmax(Hmax), Hmax.shape)
            if Hmax[r, t] < threshold:
                break
            peaks.append((r, t))
            Hmax[r, t] = 0
    return np.array(peaks).reshape((-1, 2))
