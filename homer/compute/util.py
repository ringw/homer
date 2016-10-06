import tensorflow as tf

def naive_bincount(lst):
  """Inefficient implementation of numpy's bincount."""
  pre_list = tf.cast(tf.range(0, limit=tf.reduce_max(lst)+1), lst.dtype)
  enhanced_list = tf.concat(0, [pre_list, lst])
  unused_y, unused_idx, counts = tf.unique_with_counts(enhanced_list)
  # Remove the extra elements that were in pre_list.
  return counts - 1

def find_nonzero(t, axis=0, first=True):
  bool_array = tf.cast(t, tf.boolean)
  locs = tf.where(bool_array)
  ones = tf.cast(bool_array, tf.int32)
  cumsum = tf.cumsum(t, axis=axis, reverse=not first)
  slicer = (slice(None),) * axis + (slice(1, None), Ellipsis)
