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

def hough_line_kernel(img, rhores, numrho, thetas, num_workers=32):
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
    logging.info("hough_line took " + str((e.profile.end - e.profile.start)
                                          / 10.0**9))
    return bins

rotate_prg = cl.Program(cx, open("opencl/rotate.cl").read()).build()
rotate_prg.rotate_image.set_scalar_arg_dtypes([
    None, # input image
    np.float32, # cos(theta)
    np.float32, # sin(theta)
    None, # output image
])

def rotate_kernel(img, theta):
    new_img = cla.zeros_like(img)
    rotate_prg.rotate_image(q, (img.shape[1], img.shape[0]),
                               (16, 8),
                               img.data,
                               np.cos(theta).astype(np.float32),
                               np.sin(theta).astype(np.float32),
                               new_img.data).wait()
    return new_img
