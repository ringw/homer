import tensorflow as tf

def global_runhist(page):
  t = page.tensor
  run_locs = tf.where(t[1:, :] ^ t[:-1, :])
  signed_img = tf.cast(t, tf.int32) - tf.cast(~t, tf.int32) # prob. inefficient
  run_sums = tf.cumsum(signed_img, axis=0)
  run_values = tf.gather_nd(run_sums, run_locs[1:])
  run_initials = tf.gather_nd(run_sums,
      tf.concat(1, [run_locs[:-1, 0:1], run_locs[:-1, 1:2] + 1]))
  # Don't count the first and last run of the column...
  y, idx, count = tf.unique_with_counts(run_values - run_initials)
  return count[:50]
