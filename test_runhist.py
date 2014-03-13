import pyopencl as cl
import pyopencl.array as cla
import numpy as np


W = H = 4096

localH = 128
localW = 1

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
  temp_runstart = cl.LocalMemory(localH*localW)
  temp_runs = cl.LocalMemory(localH*localW*8*4)
  h = cla.zeros(q, (2, 256), np.int32)
  all_runs = cla.zeros(q, (H, W), np.int32)

  prg = cl.Program(cx, open("runhist.cl").read()).build()

  return prg.runhist(q, (H, W/8), (localH, localW), dimg.data, temp_runstart, temp_runs, h.data, all_runs.data), h, all_runs

e, bins, runs = run()
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
#plot(B[0], 'r')
#plot(B[1], 'g')
#ylim([0,10000])
#xlim([0,30])
#show()

r = runs.get()
print r
imshow(r)
colorbar()
show()
