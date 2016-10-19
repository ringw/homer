import numpy as np
import scipy.ndimage
import tensorflow as tf
from homer.page import Page
from homer import util

MAX_SIZE = 4096

def get_rotated_page(page):
  angle, square_img = get_angle(page)
  rotated, = tf.py_func(_rotate_image, [page.image, -angle], [square_img.dtype])
  rotated.set_shape(page.image.get_shape())
  page = Page(rotated)
  page.angle = angle
  return page

def get_angle(page):
  img = tf.cast(page.image, tf.float32)
  square = get_square(img)
  f = tf.complex_abs(tf.fft2d(tf.cast(square, tf.complex64))[:MAX_SIZE//2, :])
  x_arr = (
      tf.cast(tf.concat(0,
                        [tf.range(MAX_SIZE // 2),
                         tf.range(1, MAX_SIZE // 2 + 1)[::-1]]),
              tf.float32))[None, :]
  y_arr = tf.cast(tf.range(MAX_SIZE // 2), tf.float32)[:, None]
  f = tf.select(x_arr * x_arr + y_arr * y_arr < 32 * 32, tf.zeros_like(f), f)
  m = tf.argmax(tf.reshape(f, [-1]), dimension=0)
  x = tf.cast((m + MAX_SIZE // 4) % (MAX_SIZE // 2) - (MAX_SIZE // 4), tf.float32)
  y = tf.cast(tf.floordiv(m, MAX_SIZE // 2), tf.float32)
  return(tf.cond(
      y > 0, lambda: tf.atan(x / y), lambda: tf.constant(np.nan, tf.float32)),
      square)

def get_square(img, max_size=MAX_SIZE):
  old_shape = tf.shape(img)
  new_shape, = tf.py_func(
      _resized_shape, [old_shape, max_size], [old_shape.dtype])
  resized = util.scale(img, new_shape)
  return pad_square(resized, MAX_SIZE)

def pad_square(img, max_size=MAX_SIZE):
  shape = tf.shape(img)
  padding, = tf.py_func(_get_padding, [shape, max_size], [shape.dtype])
  padded = tf.pad(255 - img, padding)
  return 255 - padded

def _resized_shape(shape, max_size):
  if shape[0] > shape[1]:
    return np.array([max_size, shape[1] * float(max_size) / shape[0]],
                    dtype=shape.dtype)
  else:
    return np.array([shape[0] * float(max_size) / shape[1], shape[1]],
                    dtype=shape.dtype)

def _get_padding(shape, max_size):
  return np.array([[0, max_size - shape[0]], [0, max_size - shape[1]]],
                  dtype=shape.dtype)

def _rotate_image(img, angle):
  # TODO: Add a GPU rotation op.
  if np.isnan(angle):
    return img
  else:
    return scipy.ndimage.interpolation.rotate(
        img, np.rad2deg(angle), reshape=False, mode='constant', cval=255.0)
