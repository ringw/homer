import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import logging

cx = cl.create_some_context()
q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)

from os import path
OPENCL = path.join(path.dirname(__file__), "../opencl")

hough_line_prg = cl.Program(cx, open(OPENCL + "/hough_line.cl").read()).build()
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
hough_line_prg.hough_lineseg.set_scalar_arg_dtypes([
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

def hough_line_kernel(img, rhores, numrho, thetas, num_workers=32):
    cos_thetas = cla.to_device(q, np.cos(thetas).astype(np.float32))
    sin_thetas = cla.to_device(q, np.sin(thetas).astype(np.float32))
    bins = cla.zeros(q, (len(thetas), numrho), np.float32)
    temp = cl.LocalMemory(4 * num_workers)
    hough_line_prg.hough_line(q, (int(numrho * num_workers), len(thetas)),
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
    temp = cl.LocalMemory(img.shape[0]*8 + img.shape[1])
    hough_line_prg.hough_lineseg(q, (len(rhos),), (1,),
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

rotate_prg = cl.Program(cx, open(OPENCL + "/rotate.cl").read()).build()
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

runhist_prg = cl.Program(cx, open(OPENCL + "/runhist.cl").read()).build()
def runhist_kernel(img):
    light = cla.zeros(q, 64, np.uint32)
    dark = cla.zeros(q, 64, np.uint32)
    runhist_prg.runhist(q, (img.shape[0], img.shape[1]), (8, 8),
                           img.data, light.data, dark.data).wait()
    return light, dark

staffpoints_prg = cl.Program(cx, open(OPENCL + "/staffpoints.cl").read()).build()
staffpoints_prg.staffpoints.set_scalar_arg_dtypes([
    None, # input image
    np.uint32, # staff_dist
    None # output image
])
def staffpoints_kernel(img, dist):
    staff = cla.zeros_like(img)
    staffpoints_prg.staffpoints(q, img.shape[::-1], (8, 8),
                                img.data, np.uint32(dist), staff.data).wait()
    return staff

maximum_filter_prg = cl.Program(cx, open(OPENCL + "/maximum_filter.cl").read())\
                       .build()
def maximum_filter_kernel(img):
    maximum = cla.zeros_like(img)
    maximum_filter_prg.maximum_filter(q, map(int, img.shape[::-1]), (1, 1),
                                         img.data, maximum.data).wait()
    return maximum

taxicab_distance_prg = cl.Program(cx, open(OPENCL + "/taxicab_distance.cl")
                                        .read()).build()
def distance_transform_kernel(img, numiters=64):
    for i in xrange(numiters):
        e = taxicab_distance_prg.taxicab_distance_step(q, img.shape[::-1],
                                                          (16, 32),
                                                          img.data)
    e.wait()

boundary_prg = cl.Program(cx, open(OPENCL + "/boundary.cl").read()).build()
boundary_prg.boundary_cost.set_scalar_arg_dtypes([
    None, # float distance transform
    np.uint32, # image width
    np.uint32, np.uint32, np.uint32, # y0, ystep, numy
    np.uint32, np.uint32, np.uint32, # x0, xstep, numx
    None # output costs
])
def boundary_cost_kernel(dist, y0, ystep, y1, x0, xstep, x1):
    numy = int(y1 - y0) // ystep
    numx = int(x1 - x0) // xstep
    costs = cla.zeros(q, (numx, numy, numy), np.float32)
    boundary_prg.boundary_cost(q, (numx, numy, numy), (1, 1, 1),
                                  dist.data,
                                  np.uint32(dist.shape[0]),
                                  np.uint32(y0),
                                  np.uint32(ystep),
                                  np.uint32(numy),
                                  np.uint32(x0),
                                  np.uint32(xstep),
                                  np.uint32(numx),
                                  costs.data).wait()
    return costs
