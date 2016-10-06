import tensorflow as tf

def roll(arr, n):
  padding_shape = tf.concat(0, [[abs(n)], arr.get_shape()[1:]])
  padding = tf.zeros(padding_shape, arr.dtype)
  arr_npositive = tf.concat(0, [padding, arr[:-n]])
  arr_nnegative = tf.concat(0, [arr[-n:], padding])
  return tf.cond(n > 0, lambda: arr_npositive, lambda: arr_nnegative)
