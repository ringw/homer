import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import logging

cx = cl.create_some_context()
q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)

hough_line_prg = cl.Program(cx, open("opencl/hough_line.cl").read()).build()
hough_line_prg.hough_line.set_scalar_arg_dtypes([
    None, # input image
    np.uint32, # image_w
    np.uint32, # image_h
    np.int32, # rhores
    None, # cos(theta)
    None, # sin(theta)
    None, # LocalMemory of size 4*local_size
    None, # output bins int32 of size (len(theta), numrho)
])

def hough_line(img, rhores, numrho, thetas, num_workers=32):
    cos_thetas = cla.to_device(q, np.cos(thetas).astype(np.float32))
    sin_thetas = cla.to_device(q, np.sin(thetas).astype(np.float32))
    bins = cla.zeros(q, (len(thetas), numrho), np.float32)
    temp = cl.LocalMemory(4 * num_workers)
    e = hough_line_prg.hough_line(q, (numrho * num_workers, len(thetas)),
                                     (num_workers, 1),
                                     img.data,
                                     np.uint32(img.shape[1]),
                                     np.uint32(img.shape[0]),
                                     np.uint32(rhores),
                                     cos_thetas.data, sin_thetas.data,
                                     temp,
                                     bins.data)
    e.wait()
    logging.warn("hough_line took " + str((e.profile.end - e.profile.start) / 10.0**9))
    return bins
