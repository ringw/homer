import pyopencl as cl
import pyopencl.array as cla
import numpy as np

cx = cl.create_some_context()
q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)

W = H = 4096

localH = 128
localW = 1

def run():
  import pyopencl as cl
  import pyopencl.array as cla
  q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)
  img = np.zeros((H, W), np.uint8)
  import moonshine
  data = moonshine.open('samples/sonata.png')[0].im
  img[:data.shape[0], :data.shape[1]] = data != 0
  imgbits = np.packbits(img)
  dimg = cla.to_device(q, imgbits)
  temp = cl.LocalMemory(8*H*localW)
  rl = cla.zeros(q, (H, W), np.uint8)

  prg = cl.Program(cx, open("runlength.cl").read()).build()

  return prg.runlength(q, (H, W/8), (localH, localW), dimg.data, temp, rl.data), rl

e, bins = run()
e.wait()

print "OpenCL:", float(e.profile.end - e.profile.start) / 1000000000

from pylab import *
B = bins.get()
#imshow(B)
#xlim([200, 600])
#colorbar()
#plot(B[3], 'g')
#plot(B[4], 'r')
#plot(img.sum(1)[:2048], 'r')
imshow(B)
colorbar()
show()
