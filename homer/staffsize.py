import numba
import numpy as np
import tensorflow as tf

def single_staffdist(page):
  staffdist, = tf.py_func(_single_staffdist, [page.tensor], [tf.int32])
  page.staffdist = staffdist
  return staffdist

def _single_staffdist(img):
  hist = runlength_pairs(img)
  peak = np.int32(np.argmax(hist))
  peak_val = hist[peak]
  hist[max(0, peak - 3):min(len(hist), peak + 4)] = 0
  return peak if hist.max() * 10 < peak_val else np.int32(-1)

@numba.jit
def runlength_pairs(img):
  img = (img != 0).astype(np.int32)
  pair_hist = np.zeros(128, np.int32)
  for col in range(img.shape[1]):
    last_run = -1
    cur_run = -1
    last_pixel = -1
    for row in range(img.shape[0]):
      if img[row, col] != last_pixel:
        last_pixel = img[row, col]
        if last_run > 0:
          run_pair = last_run + cur_run
          if run_pair < 128:
            pair_hist[run_pair] += 1
        last_run = cur_run
        cur_run = 1
      else:
        cur_run += 1
  return pair_hist
