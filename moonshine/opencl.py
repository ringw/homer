import pyopencl as cl
import pyopencl.array as cla
import numpy as np

cx = cl.create_some_context()
q = cl.CommandQueue(cx)

hough_line = cl.Program(cx, open("opencl/hough_line.cl").read()).build()
hough_line.hough_line.set_scalar_arg_dtypes([
    None, # input image
    None, # tan(theta) along axis 2
    np.int32, # rhores
    np.int32, # numrho
    None, # LocalMemory of size 4*numrho
    None, # output bins int32 of size (len(theta), numrho)
])
