from homer import util, compute
import functools
import numpy as np
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
    results = tf.py_func(_join_staves, [page.staff_dist] + staff_sections,
                         [staff_sections[0].dtype])
    return results

def _join_staves(staff_dist, *sections):
  staff_dict = dict((s, np.array([s])) for s in sections[0])
  for i, cur_staves in enumerate(sections[1:]):
    last_staves = np.sort(np.fromiter(staff_dict.keys(), int))
    dist = np.abs(last_staves[None, :] - cur_staves[:, None])
    did_match = dist.min(axis=1) < staff_dist

    new_matches = dict()
    
    matching_staves = cur_staves[did_match]
    matches = np.argmin(dist[did_match, :], axis=1)
    matches, idx = np.unique(matches, return_index=True)
    matching_staves = matching_staves[idx]
    for staff_ind, new_point in zip(matches, matching_staves):
      prev_staff = staff_dict[last_staves[staff_ind]]
      new_matches[new_point] = np.concatenate([prev_staff, [new_point]])
    
    non_matches = cur_staves[~did_match]
    for non_match in non_matches:
      new_matches[non_match] = np.asarray([-non_match] * (i + 1) + [non_match])
    
    skipped = set(staff_dict.keys()).difference(s[-2] for s in new_matches.values())
    for s in skipped:
      new_matches[s] = np.concatenate([staff_dict[s], [-s]])
    staff_dict = new_matches
  return np.asarray([staff_dict[s] for s in np.sort(np.fromiter(staff_dict.keys(), int))])


def staff_center_filter(page):
  img = page.tensor
  compute.get_staffsize(page)
  return functools.reduce(tf.logical_and,
      [util.roll(img, -page.staff_dist*2),
       util.roll(img, -page.staff_dist),
       img,
       util.roll(img, page.staff_dist),
       util.roll(img, page.staff_dist*2),
       #functools.reduce(tf.add([
       ~(util.roll(img, -3) | util.roll(img, 3)))
