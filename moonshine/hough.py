from .opencl import *
from .cl_util import maximum_filter_kernel
from pyopencl.algorithm import RadixSort
from pyopencl.scan import GenericScanKernel
import numpy as np
import logging
logger = logging.getLogger('hough')

prg = build_program("hough")
prg.hough_line.set_scalar_arg_dtypes([
    None, # input image
    np.uint32, # image_w
    np.uint32, # image_h
    np.int32, # rhores
    None, # cos(theta)
    None, # sin(theta)
    None, # LocalMemory of size 4*local_size
    None, # output bins int32 of size (len(theta), numrho)
])
prg.hough_lineseg.set_scalar_arg_dtypes([
    None, # input image
    np.uint32, # image_w
    np.uint32, # image_h
    None, # rho values
    np.uint32, # rhores
    None, # cos(theta)
    None, # sin(theta)
    np.uint32, # max_gap
    None, # LocalMemory of size >= 4 * sqrt((8*image_w)^2 + image_h^2)
    None, # output line segments shape (numlines, 4)
])
prg.can_join_segments.set_scalar_arg_dtypes([
    None, None, np.uint32
])
uint4 = cl.tools.get_or_register_dtype('uint4')

def hough_line_kernel(img, rhores, numrho, thetas, num_workers=32):
    cos_thetas = cla.to_device(q, np.cos(thetas).astype(np.float32))
    sin_thetas = cla.to_device(q, np.sin(thetas).astype(np.float32))
    bins = cla.zeros(q, (len(thetas), numrho), np.float32)
    temp = cl.LocalMemory(4 * num_workers)
    prg.hough_line(q, (int(numrho * num_workers), len(thetas)),
                                 (num_workers, 1),
                                 img.data,
                                 np.uint32(img.shape[1]),
                                 np.uint32(img.shape[0]),
                                 np.uint32(rhores),
                                 cos_thetas.data, sin_thetas.data,
                                 temp,
                                 bins.data).wait()
    return bins

def hough_lineseg_kernel(img, rhos, thetas, rhores=1, max_gap=0):
    device_rhos = cla.to_device(q, rhos.astype(np.uint32))
    cos_thetas = cla.to_device(q, np.cos(thetas).astype(np.float32))
    sin_thetas = cla.to_device(q, np.sin(thetas).astype(np.float32))
    segments = cla.zeros(q, (len(rhos), 4), np.uint32)
    temp = cl.LocalMemory(img.shape[0] + img.shape[1]/8) # bit-packed
    prg.hough_lineseg(q, (len(rhos),), (1,),
                                    img.data,
                                    np.uint32(img.shape[1]),
                                    np.uint32(img.shape[0]),
                                    device_rhos.data,
                                    np.uint32(rhores),
                                    cos_thetas.data, sin_thetas.data,
                                    temp,
                                    np.uint32(max_gap),
                                    segments.data).wait()
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

# The PyOpenCL sort kernel fails compiling for my old GPU
# (may be an OpenCL 1.0 compatibility issue)
def host_sort_segments(segments):
    segments = segments.get().view(np.uint32).reshape((-1, 4))
    segments = segments[np.argsort(segments[:,2])] # sort by y0
    return [cla.to_device(q, segments)], None
try:
    sort_segments = RadixSort(cx, "__global const uint4 *segments",
                                  "segments[i].s2", # sort by y0
                                  ["segments"],
                                  key_dtype=np.uint32)
except cl.RuntimeError:
    sort_segments = host_sort_segments

cumsum = GenericScanKernel(cx, np.int32,
                               arguments="""__global const int *can_join,
                                            __global int *labels""",
                               input_expr="can_join[i]",
                               scan_expr="a+b", neutral="0",
                               output_statement="""
                                    labels[i] = item;
                               """)

def hough_paths(segments, line_dist=40):
    # View segments as a 1D structured array
    seg_struct = segments.view(uint4).reshape(-1)
    segments, _ = sort_segments(cla.to_device(q, seg_struct))
    segments = segments[0].view(np.uint32).reshape((seg_struct.shape[0], 4))
    can_join = cla.zeros(q, segments.shape[0], np.uint32)
    prg.can_join_segments(q, segments.shape[:1], (1,),
                             segments.data, can_join.data,
                             np.uint32(line_dist))
    labels = cla.zeros(q, segments.shape[0], np.uint32)
    cumsum(can_join, labels)
    num_labels = int(labels[labels.shape[0]-1].get().item()) + 1
    longest_seg_inds = cla.empty(q, num_labels, np.uint32)
    prg.assign_segments(q, (segments.shape[0],), (1,),
                           segments.data, labels.data,
                           longest_seg_inds.data)
    longest_segs = cla.empty(q, (num_labels, 4), np.uint32)
    prg.copy_chosen_segments(q, (num_labels,), (1,),
                                segments.data,
                                longest_seg_inds.data,
                                longest_segs.data)
    return longest_segs.get()
