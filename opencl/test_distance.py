import pyopencl as cl
import pyopencl.array as cla
import numpy as np


W = H = 4096
sample = 2

localH = 4
localW = 4

def run():
  import pyopencl as cl
  import pyopencl.array as cla
  cx = cl.create_some_context()
  q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)
  img = np.zeros((H, W), np.uint8)
  import moonshine
  data = moonshine.open('samples/sonata.png')[0].im
  img[:data.shape[0], :data.shape[1]] = data != 0
  dists = np.where(img, 0, 32).astype(np.int32)
  dists = cla.to_device(q, dists)

  prg = cl.Program(cx, open("taxicab_distance.cl").read()).build()

  events = []
  for i in xrange(32):
    events.append(prg.taxicab_distance_step(q, (W/8, H), (localW, localH), dists.data))
  return events, dists

es, dists = run()
es[-1].wait()

print "OpenCL:", float(es[-1].profile.end - es[0].profile.start) / 1000000000

from pylab import *
D = dists.get()
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

imshow(D)
colorbar()
show()
