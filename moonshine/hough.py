from .gpu import *
from .cl_util import maximum_filter_kernel
import pyopencl
from reikna.core import Type
import numpy as np
import logging
logger = logging.getLogger('hough')

prg = build_program("hough")

int4 = np.dtype('i4,i4,i4,i4')

def hough_line_kernel(img, rhores, numrho, thetas, num_workers=32):
    cos_thetas = thr.to_device(np.cos(thetas).astype(np.float32))
    sin_thetas = thr.to_device(np.sin(thetas).astype(np.float32))
    bins = thr.empty_like(Type(np.float32, (len(thetas), numrho)))
    bins[:] = 0
    temp = pyopencl.LocalMemory(4 * num_workers)
    prg.hough_line(img.data,
                       np.int32(img.shape[1]),
                       np.int32(img.shape[0]),
                       np.int32(rhores),
                       cos_thetas, sin_thetas,
                       temp,
                       bins,
                       global_size=(int(numrho * num_workers), len(thetas)),
                       local_size=(num_workers, 1))
    return bins

def hough_lineseg_kernel(img, rhos, thetas, rhores=1, max_gap=0):
    device_rhos = thr.to_device(rhos.astype(np.int32))
    cos_thetas = thr.to_device(np.cos(thetas).astype(np.float32))
    sin_thetas = thr.to_device(np.sin(thetas).astype(np.float32))
    segments = thr.empty_like(Type(np.int32, (len(rhos), 4)))
    segments[:] = 0
    temp = pyopencl.LocalMemory(img.shape[0] + img.shape[1]/8) # bit-packed
    prg.hough_lineseg(img,
                      np.int32(img.shape[1]),
                      np.int32(img.shape[0]),
                      device_rhos,
                      np.int32(rhores),
                      cos_thetas, sin_thetas,
                      temp,
                      np.int32(max_gap),
                      segments,
                      global_size=(len(rhos),),
                      local_size=(1,))
    return segments

def houghpeaks(H, npeaks=2000, thresh=1.0, invalidate=(1, 1)):
    Hmax = maximum_filter_kernel(H).get()
    peaks = []
    for i in xrange(npeaks):
        r, t = np.unravel_index(np.argmax(Hmax), Hmax.shape)
        if Hmax[r, t] < thresh:
            break
        peaks.append((r, t))
        rmin = max(0, r - (invalidate[0] // 2))
        rmax = min(H.shape[0], r - (-invalidate[0] // 2))
        tmin = max(0, t - (invalidate[1] // 2))
        tmax = min(H.shape[1], t - (-invalidate[1] // 2))
        Hmax[rmin:rmax, tmin:tmax] = 0
    if len(peaks) < npeaks:
        logger.debug("houghpeaks returned %d peaks", len(peaks))
    return np.array(peaks).reshape((-1, 2))

# Reikna doesn't have a sort function?
def sort_segments(segments):
    segments = segments.get().view(np.int32).reshape((-1, 4))
    segments = segments[np.argsort(segments[:,2])] # sort by y0
    return [thr.to_device(segments)], None

#cumsum = GenericScanKernel(cx, np.int32,
#                               arguments="""__global const int *can_join,
#                                            __global int *labels""",
#                               input_expr="can_join[i]",
#                               scan_expr="a+b", neutral="0",
#                               output_statement="""
#                                    labels[i] = item;
#                               """)
def cumsum(arr, output):
    cs = np.cumsum(arr.get()).astype(np.int32)
    output[:] = cs

def hough_paths(segments, line_dist=40):
    # View segments as a 1D structured array
    seg_struct = segments.ravel().astype(np.int32).view(int4).reshape(-1)
    segments, _ = sort_segments(thr.to_device(seg_struct))
    segments = segments[0].view(np.int32).reshape((seg_struct.shape[0], 4))
    can_join = thr.empty_like(Type(np.int32, segments.shape[0]))
    can_join[:] = 0
    prg.can_join_segments(segments, can_join,
                          np.int32(line_dist),
                          global_size=segments.shape[:1],
                          local_size=(1,),)
    labels = thr.empty_like(can_join)
    cumsum(can_join, labels)
    num_labels = int(labels[labels.shape[0]-1].get().item()) + 1
    longest_seg_inds = thr.empty_like(Type(np.int32, num_labels))
    longest_seg_inds[:] = -1
    prg.assign_segments(segments.data, labels.data,
                        longest_seg_inds.data,
                        global_size=(segments.shape[0],),
                        local_size=(1,))
    longest_segs = thr.empty_like(Type(np.int32, (num_labels, 4)))
    prg.copy_chosen_segments(segments.data,
                             longest_seg_inds.data,
                             longest_segs.data,
                             global_size=(num_labels,),
                             local_size=(1,))
    return longest_segs.get()
