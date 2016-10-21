import tensorflow as tf

class Page(object):
  image = None
  threshold = 127 # TODO: thresholding
  tensor = None
  staff_space = None
  staff_thick = None
  staff_dist = None
  def __init__(self, image):
    image = tf.cast(image, tf.float32)
    image = tf.cond(
        tf.shape(tf.shape(image))[0] > 2, lambda: image[:, :, 0], lambda: image)
    self.tensor = image < self.threshold
    self.is_flipped = (tf.reduce_sum(
        tf.reshape(tf.cast(self.tensor, tf.int64), [-1]))
        > tf.cast(tf.cast(tf.reduce_prod(tf.shape(self.tensor)), tf.float32)
                  * 0.5, tf.int64))
    self.image = tf.cond(self.is_flipped, lambda: 255 - image, lambda: image)
    self.tensor = tf.cond(
        self.is_flipped, lambda: ~self.tensor, lambda: self.tensor)

  @classmethod
  def for_path(self, path):
    tf_bytes = tf.constant(open(path, 'rb').read())
    image = tf.image.decode_png(tf_bytes, channels=1, dtype=tf.uint8)
    return Page(image)
