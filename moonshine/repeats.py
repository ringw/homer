# For each staff, look for dots in the correct position before or after
# each barline.

from . import bitimage, components
from .gpu import *
import numpy as np

def staff_repeats(page, staff_num):
    img = page.staves.extract_staff(staff_num)
    # Need a byte per pixel for components
    byteimg = thr.to_device(bitimage.as_hostimage(img))
    # classes are all 1
    classes, bounds, num_pixels = components.get_components(byteimg)
    bounds = bounds.get()
    num_pixels = num_pixels.get()
    # Assume dots are an ellipse circumscribed in the bounds.
    # The "roundness" score by comparing actual pixel count
    # to perfect ellipse area
    width = bounds[:, 1] - bounds[:, 0] + 1
    height = bounds[:, 3] - bounds[:, 2] + 1
    sd = page.staff_dist
    repeat_dot_size = ((sd/3 <= width) & (width <= sd*2/3)
                       & (sd/3 <= height) & (height <= sd*2/3))
    ellipse_area = np.pi * width * height / 4.0
    roundness_score = 1.0 / (np.abs(num_pixels - ellipse_area) / ellipse_area)
    potential_dots = bounds[(roundness_score >= 5) & repeat_dot_size]
    return potential_dots
