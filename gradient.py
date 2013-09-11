import numpy as np
from scipy.ndimage.filters import convolve, gaussian_filter1d

# Return angle and magnitude
def gaussian(im, sigma=3.0):
  SHAPE = (2,) + im.shape
  gaussians = np.zeros(SHAPE, dtype=np.double)
  gaussian_filter1d(im, sigma, mode='constant', axis=0, order=1, output=gaussians[0])
  gaussian_filter1d(im, sigma, mode='constant', axis=1, order=1, output=gaussians[1])
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
    pass
