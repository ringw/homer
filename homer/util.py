import numpy as np
import tensorflow as tf

def roll(arr, n):
  if isinstance(n, tf.Tensor):
    n = tf.cast(n, tf.int64)
  else:
    n = tf.constant(n, tf.int64)
  padding_shape = tf.concat(0, [[tf.cast(abs(n), tf.int32)], tf.shape(arr)[1:]])
  padding = tf.fill(padding_shape, tf.zeros((), dtype=arr.dtype))
  negn = tf.cast(-n, tf.int32)
  arr_npositive = tf.concat(0, [padding, arr[:negn]])
  arr_nnegative = tf.concat(0, [arr[negn:], padding])
  return tf.cond(n > 0, lambda: arr_npositive, lambda: arr_nnegative)

def peaks(arr, window_size, min=np.int32(0)):
  return tf.py_func(_peaks, [arr, window_size, min], [tf.int64])[0]

def _peaks(arr, window_size, min):
  assert len(arr.shape) == 1
  is_max = np.empty_like(arr, dtype=bool)
  is_max[:window_size // 2] = False
  for i in range(window_size // 2, arr.size - (window_size // 2)):
    is_max[i] = (
        np.max(arr[i - window_size // 2:i + window_size // 2]) == arr[i])
  is_max[-(window_size // 2):] = False
  is_max &= arr >= min

  peaks = np.empty_like(arr, dtype=np.int64)
  num_peaks = 0
  for i in range(arr.size):
    if ((~is_max[i-1] and is_max[i] and ~is_max[i+1])
        or (~is_max[i-1] and is_max[i] and is_max[i+1] and ~is_max[i+2])
        or (~is_max[i-2] and is_max[i-1] and is_max[i] and is_max[i+1]
            and ~is_max[i+2])):
      peaks[num_peaks] = i
      num_peaks += 1
  return peaks[:num_peaks]

def scale(image, shape):
  dtype = image.dtype
  image = tf.cast(image, tf.float32)
  image = tf.image.resize_bicubic(
      image[None, :, :, None], shape)[0, :, :, 0]
  return tf.cast(tf.clip_by_value(image, 0.0, 255.0), dtype)
