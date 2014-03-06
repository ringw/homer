import pyopencl as cl
import pyopencl.array as cla
import numpy as np

cx = cl.create_some_context()
q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)

W = H = 4096
theta = -0.05
sintheta = np.sin(theta).astype(np.float32)
costheta = np.cos(theta).astype(np.float32)

prg = cl.Program(cx, open("rotate.cl").read()).build()
prg.rotate_image.set_scalar_arg_dtypes([None, np.int32, np.int32, None])

def run():
  import pyopencl as cl
  import pyopencl.array as cla
  q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)
  img = np.zeros((H, W), np.uint8)
  import moonshine
  data = moonshine.open('samples/chopin.pdf')[0].im
  img[:data.shape[0], :data.shape[1]] = data != 0
  imgbits = np.packbits(img)
  dimg = cla.to_device(q, imgbits)
  dout = cla.zeros(q, dimg.shape, np.uint8)
  #temp = cl.LocalMemory(4*numrho)

  return prg.rotate_image(q, (W/8, H), (8, 32), dimg.data, sintheta, costheta, dout.data), dout

e, bins = run()
e.wait()

print "OpenCL:", float(e.profile.end - e.profile.start) / 1000000000

from pylab import *
B = bins.get()
#imshow(B)
#xlim([200, 600])
#colorbar()
unpack = np.unpackbits(B)
print unpack.shape
imshow(unpack.reshape(H, W))
#plot(img.sum(1)[:2048], 'r')
show()
