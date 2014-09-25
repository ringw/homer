from ..gpu import *
from .. import util, settings
import numpy as np
import logging

prg = build_program("runhist")

def dark_runs(img):
    dark_run = thr.empty_like(Type(np.int32, 64))
    dark_run.fill(0)
    prg.dark_hist(img, dark_run, global_size=img.shape)
    return dark_run.get()

def light_runs(page, img=None):
    if img is None:
        img = page.img
    # Find light run lengths, filtering using staff_thick
    light_run = thr.empty_like(Type(np.int32, 64))
    light_run.fill(0)
    prg.light_hist(img, np.int32(page.staff_thick), light_run,
                   global_size=img.shape)
    return light_run.get()

def staffsize(page, img=None):
    if img is None:
        img = page.img
    dark_run = dark_runs(img)

    # Assume uniform staff thickness
    staff_thick = np.argmax(dark_run)
    if staff_thick > 10:
        logging.warn("Unreasonable staff_thick value: " + str(staff_thick))
        page.staff_thick = page.staff_space = page.staff_dist = None
        return None
    else:
        page.staff_thick = int(staff_thick)

    light_run = light_runs(page, img)
    # Multiple staff space sizes are possible for different instruments
    diff = np.diff(light_run)
    is_max = (diff[:-1] > 0) & (diff[1:] < 0)
    thresh = light_run[1:-1] > (light_run.max() / 10)
    space_vals = np.where(is_max & thresh)[0] + 1
    space_vals = space_vals[staff_thick < space_vals]
    # Sort by most dominant staves in piece
    # We may detect a small staff with editor's notes, etc. which should be
    # detected after the actual music staves
    space_vals = space_vals[np.argsort(-light_run[space_vals])]
    # If we assume uniform staff size, then only keep the most dominant size
    if settings.SINGLE_STAFF_SIZE:
        space_vals = space_vals[0:1]
    if len(space_vals) == 0:
        logging.warn("No staves detected")
        staff_space = None
        staff_dist = None
    elif len(space_vals) == 1:
        staff_space = int(space_vals[0])
        staff_dist = int(staff_space + staff_thick)
    else:
        staff_space = tuple(space_vals.tolist())
        staff_dist = tuple((space_vals + staff_thick).tolist())

    page.staff_space = staff_space
    page.staff_dist = staff_dist
    return staff_thick, staff_space
