import tensorflow as tf
from homer.staffsize import single_staffsize
from homer.page import Page
from homer.rotate import pad_square
from homer import util

SCALED_STAFFSIZE = 12

def get_scaled_page(page):
  staffsize = single_staffsize(page)
  shape = tf.shape(page.image)
  a = tf.cast(SCALED_STAFFSIZE, tf.float32) / tf.cast(staffsize, tf.float32)
  b = 2048.0 / tf.cast(tf.reduce_max(tf.shape(page.image)), tf.float32)
  scale = tf.cond((a < 0) | (b < a), lambda: b, lambda: a)
  new_shape = tf.cast(tf.cast(shape, tf.float32) * scale, tf.int32)
  resized = util.scale(page.image, new_shape)
  return Page(pad_square(resized, 2048))
