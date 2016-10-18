from homer.compute.staffsize import get_staffsize

SCALED_STAFFSIZE = 12

def get_scaled_page(page):
  staffsize = get_staffsize(page)
  scale = tf.cast(SCALED_STAFFSIZE, tf.float32) / tf.cast(staffsize, tf.float32)
  scale = tf.select(scale < 1.0, tf.constant(1.0), scale)
  shape = tf.shape(page.square_image)
  new_shape = tf.cast(tf.cast(shape, tf.float32) * scale, tf.int32)
  resized = tf.image.resize_bicubic(
      img[None, :, :, None], new_shape)[0, :, :, 0]
  return get_square(resized)
