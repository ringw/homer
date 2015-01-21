# An implementation of the algorithm:
# Cardoso JS, Rebelo A (2010) Robust staffline thickness and distance
# estimation in binary and gray-level music scores.
# In: Proceedings of the twentieth international conference on pattern
# recognition, pp 1856-1859

from .gpu import *
from . import util, settings
import numpy as np
import logging

prg = build_program("staffsize")

def staff_dist_hist(page, img=None):
    if img is None:
        img = page.img
    dist_runs = thr.empty_like(Type(np.int32, 64))
    dist_runs.fill(0)
    prg.staff_dist_hist(img, dist_runs, global_size=img.shape)
    return dist_runs.get()

def staff_thick_hist(page, img=None):
    if img is None:
        img = page.img
    # Find light run lengths, filtering using staff_thick
    thick_runs = thr.empty_like(Type(np.int32, 64))
    thick_runs.fill(0)
    if type(page.staff_dist) is tuple:
        staff_dist = page.staff_dist[0]
    else:
        staff_dist = page.staff_dist
    prg.staff_thick_hist(img, np.int32(staff_dist), thick_runs,
                         global_size=img.shape)
    return thick_runs.get()

def staffsize(page, img=None):
    if img is None:
        img = page.img
    dist = staff_dist_hist(page, img)

    if dist.max() < page.orig_size[1] * 10:
        logging.warn("Few runs detected, likely no staves in image")
        page.staff_thick = None
        page.staff_space = None
        page.staff_dist = None
        return page.staff_thick, page.staff_dist
    # Multiple staff sizes are possible for different instruments
    # Look for distinct peaks
    is_max = np.zeros_like(dist, bool)
    is_max[1:-1] = dist[1:-1] > np.max([dist[:-2], dist[2:]], axis=0)
    # Must be a sharp peak
    is_max[2:-2] &= dist[2:-2] > np.max([dist[:-4], dist[4:]], axis=0)*5
    dist_vals, = np.where(is_max)
    # Sort by most dominant staves in piece
    # We may detect a small staff with editor's notes, etc. which should be
    # detected after the actual music staves
    # XXX: staff_dist value is one greater than expected???
    dist_vals = dist_vals[np.argsort(-dist[dist_vals])] - 1
    # If we assume uniform staff size, then only keep the most dominant size
    if settings.SINGLE_STAFF_SIZE:
        dist_vals = dist_vals[0:1]
    if len(dist_vals) == 0:
        logging.warn("No staves detected")
        page.staff_thick = None
        page.staff_space = None
        page.staff_dist = None
        return page.staff_thick, page.staff_dist
    elif len(dist_vals) == 1:
        staff_dist = int(dist_vals[0])
    else:
        staff_dist = tuple(dist_vals.tolist())
    page.staff_dist = staff_dist

    # Assume uniform staff thickness for all staff_dist values
    thick_runs = staff_thick_hist(page, img)
    staff_thick = int(np.argmax(thick_runs))
    if staff_thick > 10:
        logging.warn("Unreasonable staff_thick value: " + str(staff_thick))
        page.staff_thick = page.staff_space = page.staff_dist = None
        return None
    else:
        page.staff_thick = staff_thick

    if type(staff_dist) is tuple:
        staff_space = tuple(sd - staff_thick for sd in staff_dist)
    else:
        staff_space = staff_dist - staff_thick

    page.staff_space = staff_space
    page.staff_dist = staff_dist
    return staff_thick, staff_space
