from .opencl import *
from . import util
import numpy as np
import logging

def staffsize(page):
    light_run_device, dark_run_device = runhist_kernel(page.img)
    light_run = light_run_device.get()
    dark_run = dark_run_device.get()

    # Assume uniform staff thickness
    staff_thick = np.argmax(dark_run)
    if staff_thick > 5:
        logging.warn("Unreasonable staff_thick value: " + str(staff_thick))

    # Multiple staff space sizes are possible for different instruments
    space = light_run > light_run.max() / 2.0
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
