import pyopencl as cl
import pyopencl.array as cla
import numpy as np


W = H = 4096

localH = 16
localW = 16

def run():
  import pyopencl as cl
  import pyopencl.array as cla
  cx = cl.create_some_context()
  q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)
  img = np.zeros((H, W), np.uint8)
  import moonshine
  data = moonshine.open('samples/sonata.png')[0].im
  img[:data.shape[0], :data.shape[1]] = data != 0
  imgbits = np.packbits(img)
  dimg = cla.to_device(q, imgbits)
  staff = cla.zeros(q, (H, W/8), np.uint8)

  prg = cl.Program(cx, open("staffpoints.cl").read()).build()
  prg.staffpoints.set_scalar_arg_dtypes([None, np.int32, None])

  return prg.staffpoints(q, (W/8, H), (localW, localH), dimg.data, np.int32(21), staff.data), staff

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
