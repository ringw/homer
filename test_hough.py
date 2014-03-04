import pyopencl as cl
import pyopencl.array as cla
import numpy as np

cx = cl.create_some_context()
q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)

W = H = 4096
rhores = 1
numrho = 2048
theta = np.tan(np.linspace(-np.pi/100, np.pi/100, 11))
dtheta = cla.to_device(q, theta.astype(np.float32))

img = np.zeros((H, W), np.uint8)
#img[np.arange(500, 520), np.arange(200, 240, 2)] = 1
import moonshine
data = moonshine.open('samples/chopin.pdf')[0].im
img[:data.shape[0], :data.shape[1]] = data != 0
dimg = cla.to_device(q, img)
temp = cl.LocalMemory(4*numrho)
bins = cla.zeros(q, (len(theta), numrho), np.int32)

prg = cl.Program(cx, open("hough_line.cl").read()).build()
prg.hough_line.set_scalar_arg_dtypes([None, None, np.int32, np.int32, None, None])

e = prg.hough_line(q, (len(theta), H, W), (1, 2, 16), dimg.data, dtheta.data, np.int32(1), np.int32(numrho), temp, bins.data)
e.wait()

print "OpenCL:", float(e.profile.end - e.profile.start) / 1000000000

from pylab import *
B = bins.get()
imshow(B)
xlim([200, 600])
colorbar()
show()
