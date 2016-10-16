from homer.util import peaks, roll
from homer.staves import base
import tensorflow as tf

class ProjectionStaffDetector(base.BaseStaffDetector):
  def get_staves(self, page):
    self.page = page
    return super().get_staves(page)

  def detect(self, img):
    proj = tf.reduce_sum(tf.cast(img, tf.int32), reduction_indices=[1])
    return peaks(proj, window_size=self.page.staff_dist)
