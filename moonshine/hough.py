from .opencl import *
from pyopencl.algorithm import RadixSort
from pyopencl.scan import GenericScanKernel
import numpy as np
import logging
logger = logging.getLogger('hough')

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
        logger.info("houghpeaks returned %d peaks", len(peaks))
    return np.array(peaks).reshape((-1, 2))

uint4 = cl.tools.get_or_register_dtype('uint4')
sort_segments = RadixSort(cx, "__global const uint4 *segments",
                              "segments[i].s2", # sort by y0
                              ["segments"],
                              key_dtype=np.uint32)
cumsum = GenericScanKernel(cx, np.int32,
                               arguments="""__global const int *can_join,
                                            __global int *labels""",
                               input_expr="can_join[i]",
                               scan_expr="a+b", neutral="0",
                               output_statement="""
                                    labels[i] = item;
                               """)
prg = cl.Program(cx, """
#define MIN(a,b) (((a)<(b)) ? (a) : (b))
#define MAX(a,b) (((a)>(b)) ? (a) : (b))

// Sort hough line segments before using.
// Set can_join to 1 if the segment should be considered part of the same
// staff or barline as the previous segment. This is prefix summed
// to get unique ids for each staff.
__kernel void can_join_segments(__global const int4 *segments,
                                __global int *can_join,
                                int threshold) {
    uint i = get_global_id(0);
    if (i == 0)
        can_join[i] = 0;
    else
        can_join[i] = abs(MIN(segments[i].s2, segments[i].s3)
                            - MAX(segments[i-1].s2, segments[i-1].s3))
                            > threshold
                    ? 1 : 0;
}

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
// Assign longest segment for each label atomically.
// Everything should be a ushort4 to allow atomic operations on long values.
__kernel void assign_segments(__global const ushort4 *segments,
                              __global const int *labels,
                              __global volatile ushort4 *longest) {
    uint i = get_global_id(0);
    ushort4 seg = segments[i];
    float4 segf = convert_float4(seg);
    // Get length of segment from dot product
    float dx = dot(segf, (float4)(-1, 1, 0, 0));
    float dy = dot(segf, (float4)(0, 0, -1, 1));
    float seg_length = dot((float2)(dx, dy), (float2)(dx, dy));
    int label = labels[i];
    union {
        ushort4 s;
        ulong l;
    } seg_u, old_u;
    seg_u.s = seg;
    do {
        ushort4 longest_seg = longest[label];
        segf = convert_float4(longest_seg);
        dx = dot(segf, (float4)(-1, 1, 0, 0));
        dy = dot(segf, (float4)(0, 0, -1, 1));
        float longest_length = dot((float2)(dx, dy), (float2)(dx, dy));
        if (longest_length >= seg_length)
            break;
        old_u.s = longest_seg;
    } while (atom_cmpxchg((__global volatile ulong *)&longest[label],
                          old_u.l, seg_u.l) != old_u.l);
}

// See if we can extend each initial Hough segment with another
// almost-parallel segment to increase its length.
__kernel void extend_lines(__global const ushort4 *segments,
                           __global const ushort4 *initial_segment,
                           __global const int *labels,
                           __global volatile ushort4 *extension) {
    uint i = get_global_id(0);
    union {
        ushort4 s;
        ulong l;
    } seg_u, old_u;
    seg_u.s = segments[i];
    float4 seg = convert_float4(seg_u.s);
    int label = labels[i];
    old_u.s = initial_segment[i];
}
""").build()
prg.can_join_segments.set_scalar_arg_dtypes([
    None, None, np.uint32
])

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
    first_segments = cla.zeros(q, (int(labels[labels.shape[0]-1].get().item() + 1), 4), np.uint16)
    prg.assign_segments(q, (segments.shape[0],), (1,),
                           segments.astype(np.uint16).data, labels.data,
                           first_segments.data)
    return first_segments.get()
