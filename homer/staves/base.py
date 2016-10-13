from homer import util, compute
import functools
import tensorflow as tf

class BaseStaffDetector(object):
  num_slices = 8

  def detect(self, img):
    raise Exception("Abstract method")

  def get_staves(self, page):
    filt = staff_center_filter(page)
    n = self.num_slices
    increment = tf.floordiv(tf.shape(filt)[1], tf.constant(n + 1, tf.int32))
    staff_sections = []
    for i in range(n):
      img = filt[:, i*increment:(i+1)*increment if i + 1 < n else None]
      staff_sections.append(self.detect(img))
    result, result2 = tf.py_func(join_staves, [page.staff_dist] + staff_sections, [staff_sections[0].dtype, staff_sections[1].dtype])
    return result, result2

def join_staves(staff_dist, *sections):
  last_staves = [sections[0]]
  for cur_staves in sections[1:]:
    dist = np.abs(last_staves[-1][None, :] - cur_staves[:, None])
    did_match = dist.min(axis=1) < staff_dist
    cur_staves = cur_staves[did_match]
    matches = np.argmin(dist[did_match, :], axis=1)
    matches, idx = np.unique(matches, return_index=True)
    cur_staves = cur_staves[idx]
    last_staves = [s[matches] for s in last_staves]
    last_staves.append(cur_staves)
    return last_staves

def staff_center_filter(page):
  img = page.tensor
  compute.get_staffsize(page)
  return functools.reduce(tf.logical_and,
      [util.roll(img, -page.staff_dist*2),
       util.roll(img, -page.staff_dist),
       img,
       util.roll(img, page.staff_dist),
       util.roll(img, page.staff_dist*2)])
