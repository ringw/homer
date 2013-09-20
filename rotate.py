from numpy import *
import Image

class RotateTask:
  def __init__(self, page):
    self.page = page
    im_string = page.im.astype(uint8).tostring()
    self.pil_im = Image.fromstring('L', page.im.shape[::-1], im_string)

  def test_angles(self, ts):
    rotated_ims = [self.pil_im.rotate(t * 180.0 / pi) for t in ts]
    all_im_string = ''.join([im.tostring() for im in rotated_ims])
    np_ims = fromstring(all_im_string, dtype=uint8).reshape((len(ts),) + self.page.im.shape)
    scores = sum(sum(np_ims, axis=-1) ** 2, axis=-1)
    return scores, np_ims

  def rotate_image(self):
    # Determine direction of rotation
    scores, _ = self.test_angles([-pi/180, 0, pi/180])
    left, right, straight = scores
    dt = 0
    if left > straight and straight > right:
      dt = -pi/180
    elif left < straight and straight < right:
      dt = pi/180
    # If needs to be rotated, determine rotation to nearest degree
    rotate_base = 0
    if dt:
      rotate_angles = arange(1, 11) * dt
      scores, _ = self.test_angles(rotate_angles)
      rotate_base = rotate_angles[argmax(scores)]
    # Determine angle to nearest 0.05 degree
    rotate_candidates = linspace(-pi/360, pi/360, 21) + rotate_base
    scores, ims = self.test_angles(rotate_candidates)
    best_angle = argmax(scores)
    print degrees(rotate_candidates[best_angle])
    return ims[best_angle]

  def process(self):
    self.page.im = self.rotate_image()
  def color_image(self):
    pass
