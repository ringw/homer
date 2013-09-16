import numpy as np
from scipy import ndimage

def rotate_profile(ts, num_x):
  ts = np.asarray(ts)
  if np.any((ts < -np.pi/4) | (ts > np.pi/4)):
    raise ValueError("Angle must be between -pi/4 and +pi/4")
  xs = np.arange(num_x)
  return np.rint(np.tan(ts)[..., None] * xs[(None,)*len(ts.shape)]).astype(int)

class RotateTask:
  def __init__(self, page):
    self.page = page

  def find_angle(self):
    ts = -1/180.0*np.pi
    profiles = rotate_profile(ts, self.page.im.shape[1])
    print profiles.shape
    ims = self.page.im[(np.arange(self.page.im.shape[0])[:, None]
                        + profiles[None, :]) % self.page.im.shape[0],
                       np.arange(self.page.im.shape[1])[None, :]]
    return ims

  def process(self):
    pass
  def color_image(self):
    pass
