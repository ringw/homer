import numpy as np
from opencl import *
from . import image, rotate, layout

PAGE_SIZE = 4096

class Page:
  def __init__(self, image_data):
    img = image.image_array(self.image_data)
    padded_img = np.zeros((PAGE_SIZE, PAGE_SIZE), np.uint8)
    padded_img[:img.shape[0], :img.shape[1]] = img
    self.bitarray = np.packbits(img)
    self.img = cla.to_device(q, self.bitarray)

  def show(self, show_tasks=True):
    import pylab
    pylab.figure()
    if show_tasks:
      for task in self.tasks:
        task.show()
    pylab.imshow(self.im != 0)
