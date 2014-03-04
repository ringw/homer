import pyopencl as cl
import pyopencl.array as cla
import numpy as np

cx = cl.create_some_context()
q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)

N = 1048576*2
axis = 8
numbins = 128

group0 = 16
group1 = 32

#vals = np.random.randint(0, numbins, N).astype(np.int16)
vals = np.arange(0, numbins).repeat(N / numbins).astype(np.int16)
dvals = cla.to_device(q, vals)
temp = cl.LocalMemory(4*numbins)
bins = cla.zeros(q, numbins, np.int32)

prg = cl.Program(cx, open("bincount.cl").read()).build()
prg.bincount.set_scalar_arg_dtypes([None, np.int32, None, None])

e = prg.bincount(q, (1, axis, N/axis), (1, 2, 32), dvals.data, np.int32(numbins), temp, bins.data)
e.wait()

print "OpenCL:", float(e.profile.end - e.profile.start) / 1000000000

import time
TIMES = 10
total = 0.0
for i in xrange(TIMES):
  start = time.time()
  npbins = np.bincount(vals)
  end = time.time()
  total += end - start

print "NumPy:", total / TIMES

clbins = bins.get()
print np.array([clbins[:len(npbins)], npbins])
print (clbins[:len(npbins)] == npbins).all()
