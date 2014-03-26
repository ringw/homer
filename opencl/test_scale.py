import pyopencl as cl
import pyopencl.array as cla
import numpy as np

cx = cl.create_some_context()
q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)

W = H = 4096
scale = 2.0
scaleW = scaleH = 2048

prg = cl.Program(cx, open("scale.cl").read()).build()
prg.scale_image.set_scalar_arg_dtypes([None, np.float32, np.uint32, np.uint32, None])

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
  dout = cla.zeros(q, (scaleH, scaleW/8), np.uint8)

  return prg.scale_image(q, (scaleW/8, scaleH), (8, 32), dimg.data, np.float32(scale), np.uint32(scaleW/8), np.uint32(scaleH), dout.data), dout

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
imshow(unpack.reshape(scaleH, scaleW))
#plot(img.sum(1)[:2048], 'r')
show()
