from ..gpu import *
from .. import util
import numpy as np
import logging

prg = build_program("runhist")

def staffsize(page):
    dark_run = thr.empty_like(Type(np.int32, 64))
    dark_run.fill(0)
    prg.dark_hist(page.img, dark_run, global_size=page.img.shape)
    dark_run = dark_run.get()

    # Assume uniform staff thickness
    staff_thick = np.argmax(dark_run)
    if staff_thick > 10:
        logging.warn("Unreasonable staff_thick value: " + str(staff_thick))
        page.staff_thick = page.staff_space = page.staff_dist = None
        return None

    # Find light run lengths, filtering using staff_thick
    light_run = thr.empty_like(dark_run)
    light_run.fill(0)
    prg.light_hist(page.img, np.int32(staff_thick), light_run,
                   global_size=page.img.shape)
    light_run = light_run.get()

    # Multiple staff space sizes are possible for different instruments
    space = light_run > light_run.max() / 5.0
    space_label, num_spaces = util.label_1d(space)
    space_vals = np.rint(util.center_of_mass_1d(space_label)).astype(int)
    space_vals = space_vals[(staff_thick * 2 < space_vals)
                            & (space_vals < 50)]
    if len(space_vals) == 0:
        logging.warn("No staves detected")
        staff_space = None
        staff_dist = None
    elif len(space_vals) == 1:
        staff_space = space_vals[0]
        staff_dist = staff_space + staff_thick
    else:
        staff_space = tuple(space_vals)
        staff_dist = tuple(space_vals + staff_thick)

    page.staff_thick = staff_thick
    page.staff_space = staff_space
    page.staff_dist = staff_dist
    return staff_thick, staff_space
