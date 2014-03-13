import pyopencl as cl
import pyopencl.array as cla
import numpy as np


W = H = 4096
sample = 2

localH = 16
localW = 16

x0, xstep, numx = 0, 100, 10
y0, ystep, numy = 500, 10, 10

def run():
  import pyopencl as cl
  import pyopencl.array as cla
  cx = cl.create_some_context()
  q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)
  img = np.zeros((H, W), np.float32)
  import moonshine
  data = moonshine.open('samples/sonata.png')[0].im
  img[:data.shape[0], :data.shape[1]] = data != 0
  dimg = cla.to_device(q, img[:H/sample, :W/sample].copy())
  costs = cla.zeros(q, (numx, numy, numy), np.float32)

  prg = cl.Program(cx, open("boundary.cl").read()).build()
  prg.boundary_cost.set_scalar_arg_dtypes([None, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, None])

  return prg.boundary_cost(q, (numx, numy, numy), (1, 1, 1), dimg.data, np.int32(W/sample), np.int32(y0), np.int32(ystep), np.int32(numy), np.int32(x0), np.int32(xstep), np.int32(numx), costs.data), costs

e, staff = run()
e.wait()

print "OpenCL:", float(e.profile.end - e.profile.start) / 1000000000

from pylab import *
S = staff.get()
#imshow(B)
#xlim([200, 600])
#colorbar()
#plot(B[3], 'g')
#plot(B[4], 'r')
#plot(img.sum(1)[:2048], 'r')
#plot(B[0], 'r')
#plot(B[1], 'g')
#ylim([0,10000])
#xlim([0,30])
#show()

imshow(np.unpackbits(S).reshape((H,-1)))
show()
