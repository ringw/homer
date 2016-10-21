import numpy as np
import tensorflow as tf
from . import util

def global_runhist(page):
  t = tf.transpose(page.tensor)
  run_starts = t[:, 1:] ^ t[:, :-1]
  run_indices = tf.where(run_starts)
  flat = tf.reshape(run_starts, [-1]) # column-major order of the original image
  segment_ids_flat = tf.cumsum(tf.cast(flat, np.int32))

  # The segment ids at the end of each row are invalid
  # (because they wrap around.)
  num_cols = tf.shape(t)[1]
  bad_ids, unused_idx = tf.unique(segment_ids_flat[num_cols - 1::num_cols])

  unique_segments, unused_idx, run_lengths = tf.unique_with_counts(
      segment_ids_flat)
  is_black_run = tf.gather_nd(t, run_indices)

  # Now remove bad_ids from everything.
  unused_good_ids, good_idx = tf.listdiff(unique_segments, bad_ids)
  run_lengths = tf.gather(run_lengths, good_idx)
  is_black_run = tf.gather(is_black_run, good_idx)

  white_runs = tf.gather_nd(run_lengths, tf.where(tf.logical_not(is_black_run)))
  black_runs = tf.gather_nd(run_lengths, tf.where(is_black_run))
  return util.bincount(white_runs), util.bincount(black_runs)

def get_staffsize(page):
  white, black = global_runhist(page)
  minval = tf.shape(page.image)[1]
  page.staff_space = single_peak(white, minval=minval)
  page.staff_thick = single_peak(black, minval=minval)
  is_valid = tf.logical_and(
      tf.greater(page.staff_space, -1), tf.greater(page.staff_thick, -1))
  page.staff_dist = tf.cond(is_valid,
      lambda: page.staff_space + page.staff_thick,
      lambda: tf.constant(-1, dtype=page.staff_space.dtype))
  return page.staff_dist

def single_peak(hist, cutoff=0.5, minval=0):
  peak = tf.argmax(hist, dimension=0)[0]
  peak_height = tf.reduce_max(hist)
  shift_before = tf.concat(concat_dim=0, values=[[0], hist[:-1]])
  shift_after = tf.concat(concat_dim=0, values=[hist[1:], [0]])
  all_peaks = tf.where(
      tf.logical_and(hist > shift_before, hist > shift_after))[:, 0]
  minor_peaks, unused_idx = tf.listdiff(all_peaks, peak[None])
  minor_heights = tf.gather(hist, minor_peaks)
  cutoff = tf.constant(cutoff, dtype=tf.float32)
  minor_max = tf.cast(tf.reduce_max(minor_heights), dtype=cutoff.dtype)
  # Create eagerly, so we don't get weird errors creating a tensor in a lambda.
  return tf.cond(
      (tf.cast(peak_height, dtype=cutoff.dtype) * cutoff > minor_max)
          & (peak_height > minval),
      lambda: peak, lambda: tf.constant(-1, dtype=peak.dtype))
