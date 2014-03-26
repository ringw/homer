import numpy as np
from opencl import *

HOUGH_NUM_THETA = 11 # should be odd
def rotate(page):
    """ Find the angle with the maximum sum of squares of each Hough bin.
        This should correspond to the strongest horizontal lines.
    """
    ts = np.linspace(-np.pi/100, np.pi/100, HOUGH_NUM_THETA)
    tan_ts = cla.to_device(q, np.tan(ts).astype(np.float32))
    bins = cla.zeros(q, (HOUGH_NUM_THETA))

class RotateTask:
  def __init__(self, page):
    self.page = page
    self.pil_im = None

  def test_angles(self, ts):
    if self.pil_im is None:
      im_string = self.page.im.astype(np.uint8).tostring()
      self.pil_im = Image.frombytes('L', self.page.im.shape[::-1], im_string)

    scores = np.empty(len(ts))
    best_score = 0
    best_im = None
    for ind, t in enumerate(ts):
      rotated = self.pil_im.rotate(t * 180.0 / np.pi)
      im_string = rotated.tobytes()
      # Score is square of vertical projection
      # If staff is lined up nearly horizontally, score will be maximized
      im = (np.fromstring(im_string, dtype=np.uint8)
              .reshape(self.page.im.shape))
      projection = im.sum(axis=-1)
      score = np.sum(projection ** 2, axis=-1)
      scores[ind] = score
      if score > best_score:
        best_score = score
        best_im = im
    return scores, best_im

  def rotate_image(self):
    # Determine direction of rotation
    scores, _ = self.test_angles([-np.pi/720, 0, np.pi/720])
    left, right, straight = scores
    dt = 0
    if left > straight and straight > right:
      dt = -np.pi/180
    elif left < straight and straight < right:
      dt = np.pi/180
    # If needs to be rotated, determine rotation to nearest degree
    rotate_base = 0
    if dt:
      rotate_angles = np.arange(11) * dt
      scores, _ = self.test_angles(rotate_angles)
      rotate_base = rotate_angles[np.argmax(scores)]
    # Determine angle to nearest 0.05 degree
    rotate_candidates = np.linspace(-np.pi/360, np.pi/360, 21) + rotate_base
    scores, im = self.test_angles(rotate_candidates)
    best_angle = np.argmax(scores)
    return im, float(rotate_candidates[best_angle])

  def process(self):
    self.page.im, self.t = self.rotate_image()
    # Force page to reload runlength encoding and spacing
    self.page.get_runlength_encoding()
    self.page.get_spacing()

  def show(self):
    pass # nothing to draw
