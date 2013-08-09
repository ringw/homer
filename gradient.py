import numpy as np
from scipy.ndimage.filters import convolve, gaussian_filter1d

# Return angle and magnitude
def gaussian(im, sigma=3.0):
  SHAPE = (2,) + im.shape
  gaussians = np.zeros(SHAPE, dtype=np.double)
  gaussian_filter1d(im, sigma, axis=0, order=1, output=gaussians[0])
  gaussian_filter1d(im, sigma, axis=1, order=1, output=gaussians[1])
  output = np.zeros(SHAPE, dtype=np.double)
  output[0] = np.arctan2(gaussians[0], gaussians[1])
  output[1] = np.sqrt(gaussians[0]**2 + gaussians[1]**2)
  return output

class GradientTask:
  def __init__(self, page):
    self.page = page
  def process(self):
    self.page.gradient = gaussian(self.page.im, sigma=self.page.staff_thick)
  def color_image(self):
    return
    #import matplotlib
    import matplotlib.pyplot as plt
    #cmap = matplotlib.cm.jet
    #cmap.set_bad('w',1.)
    border = (self.page.im
              & (convolve(self.page.im, np.ones((3,3)), mode='constant') < 9))
    by,bx = np.where(border)
    img = np.zeros(border.shape)
    img[:] = 10
    img[by,bx] = self.gradient[0,by,bx]
    #mask = np.ma.masked_where(self.gaussian[0], mask=border)
    #plt.imshow(img[0, 500:1000, 500:1000])
    #plt.colorbar()
    #plt.show()
    h = np.histogram(self.gaussian[0].ravel(), bins=360, range=(-np.pi,np.pi))
    print h
