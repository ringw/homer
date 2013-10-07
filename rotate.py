from numpy import *
from scipy import ndimage
import Image

class RotateTask:
  def __init__(self, page):
    self.page = page
    self.pil_im = None

  def test_angles(self, ts):
    if self.pil_im is None:
      im_string = self.page.im.astype(uint8).tostring()
      self.pil_im = Image.fromstring('L', self.page.im.shape[::-1], im_string)

    rotated_ims = [self.pil_im.rotate(t * 180.0 / pi) for t in ts]
    all_im_string = ''.join([im.tostring() for im in rotated_ims])
    np_ims = fromstring(all_im_string, dtype=uint8).reshape((len(ts),) + self.page.im.shape)
    # Horizontal projection
    im_proj = sum(np_ims, axis=-1)
    scores = sum(im_proj ** 2, axis=-1)
    return scores, np_ims

  def rotate_image(self):
    # Determine direction of rotation
    scores, _ = self.test_angles([-pi/720, 0, pi/720])
    left, right, straight = scores
    dt = 0
    if left > straight and straight > right:
      dt = -pi/180
    elif left < straight and straight < right:
      dt = pi/180
    # If needs to be rotated, determine rotation to nearest degree
    rotate_base = 0
    if dt:
      rotate_angles = arange(11) * dt
      scores, _ = self.test_angles(rotate_angles)
      rotate_base = rotate_angles[argmax(scores)]
    # Determine angle to nearest 0.05 degree
    rotate_candidates = linspace(-pi/360, pi/360, 21) + rotate_base
    scores, ims = self.test_angles(rotate_candidates)
    best_angle = argmax(scores)
    return ims[best_angle], rotate_candidates[best_angle]

  def process(self):
    self.page.im, self.t = self.rotate_image()
    self.page.colored = Image.fromstring('L', (self.page.im.shape[0],
                                               self.page.im.shape[1]),
                                         (self.page.im*255).tostring()) \
                             .convert('RGBA')
    # Force page to reload runlength encoding and spacing
    self.page.get_runlength_encoding()
    self.page.get_spacing()
  def color_image(self):
    pass
