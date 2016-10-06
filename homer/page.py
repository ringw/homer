import tensorflow as tf

class Page(object):
  image = None
  threshold = 127 # TODO: thresholding
  tensor = None
  staff_space = None
  staff_thick = None
  staff_dist = None
  def __init__(self, image):
    self.image = image
    self.tensor = image[:, :, 0] < self.threshold

  @classmethod
  def for_path(self, path):
    tf_bytes = tf.constant(open(path, 'rb').read())
    image = tf.image.decode_png(tf_bytes, channels=1, dtype=tf.uint8)
    return Page(image)
