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
