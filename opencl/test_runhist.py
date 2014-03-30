import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import moonshine
from moonshine.opencl import *


W = H = 4096

localH = 128
localW = 1

def run():
  import pyopencl as cl
  import pyopencl.array as cla
  dimg = moonshine.open('samples/sonata.png')[0].img
  l = cla.zeros(q, 64, np.uint32)
  d = cla.zeros(q, 64, np.uint32)

  prg = cl.Program(cx, open("opencl/runhist.cl").read()).build()

  return prg.runhist(q, (H, W/8), (localH, localW), dimg.data, l.data, d.data), l, d

e, l, d = run()
e.wait()

print "OpenCL:", float(e.profile.end - e.profile.start) / 1000000000

from pylab import *
L = l.get()
D = d.get()

figure()
plot(L)
figure()
plot(D)
show()
