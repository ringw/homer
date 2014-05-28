# General utilities for OpenCL arrays
from .opencl import *
from pyopencl.reduction import ReductionKernel

prg = build_program(["maximum_filter", "taxicab_distance"])
def maximum_filter_kernel(img):
    maximum = cla.zeros_like(img)
    prg.maximum_filter(q, map(int, img.shape[::-1]), (1, 1),
                                         img.data, maximum.data).wait()
    return maximum

max_kernel = ReductionKernel(cx, np.float32, neutral="0.f",
                             reduce_expr="(a>b) ? a : b",
                             map_expr="x[i]",
                             arguments="__global float *x")
