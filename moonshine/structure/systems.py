# Join adjacent staves into a system if a barline connects them.
from .. import opencl, util, bitimage, hough
import numpy as np

def initialize_systems(page):
    """ Initial systems are each individual staff,
        barlines are a vertical line through the staff """
    page.systems = []
    i = 0
    for staff, barlines in zip(page.staves, page.barlines):
        x0, x1, y0, y1 = staff
        system_bars = []
        for barline_x in barlines:
            staff_y = y0 + (y1 - y0) * (barline_x - x0) / (x1 - x0)
            system_bars.append([barline_x, barline_x,
                                staff_y - page.staff_dist*2,
                                staff_y + page.staff_dist*2])
        page.systems.append(dict(barlines=np.array(system_bars, int),
                                 start=i, stop=i))
        i += 1

def verify_barlines(page, i, j, barlines):
    """ Convert each barline to a rho and theta value, then verify that
        a line exists using hough_lineseg. """
    # Get a slice of the image which includes systems i through j
    assert i <= j
    y0 = max(0, page.staves[i,[2,3]].min() - page.staff_dist*3)
    y1 = min(page.img.shape[0],
             page.staves[j,[2,3]].max() + page.staff_dist*3)
    # Gap between y0 and y1 must be a multiple of 8
    y1 += 8 - ((y1 - y0) & 0x7)
    img_slice = page.img[y0:y1].copy()
    slice_T = bitimage.transpose(img_slice)
    slice_barlines = barlines.copy()
    slice_barlines[:,[2,3]] -= y0
    slice_barlines[slice_barlines < 0] = 0
    # Equations for transposed image:
    # rho = x cos t + y sin t
    # 0 = (x1 - x0) cos t + (y1 - y0) sin t
    # tan t = - (x1 - x0) / (y1 - y0)
    t = np.arctan(-(barlines[:,1] - barlines[:,0]).astype(float)
                   / (barlines[:,3] - barlines[:,2]))
    rho = barlines[:,0] * np.cos(t) + barlines[:,2] * np.sin(t)
    rhores = page.staff_thick * 2
    rhoval = (rho / rhores).astype(int) + 1
    new_lines = hough.hough_lineseg_kernel(
                    slice_T, rhoval, t, rhores, max_gap=page.staff_dist).get()
    new_lines = new_lines[:, [2,3, 0,1]]
    new_lines[:, [2,3]] += y0
    new_lines = new_lines[(new_lines[:,2] < page.staves[j,[0,1]].max()
                                                - page.staff_dist)
                          & (new_lines[:,3] > page.staves[j,[2,3]].max()
                                                + page.staff_dist)]
    return new_lines
def try_join_system(page, i):
    """ Try joining system i with the system below it.
        Update page.systems, returning True,
        or return False if no barlines connect. """
    # Match each barline x1 in the top system to x0 in the bottom system
    system0_x1 = page.systems[i]['barlines'][:,1]
    system1_x0 = page.systems[i+1]['barlines'][:,0]
    matches = util.match(system0_x1, system1_x0)
    match1_x0 = page.systems[i+1]['barlines'][matches, 0]
    good_match = np.abs(system0_x1 - match1_x0) < page.staff_dist/2
    system0 = page.systems[i]['barlines'][good_match]
    system1 = page.systems[i+1]['barlines'][matches[good_match]]
    new_systems = np.c_[system0[:,0], system1[:,1], system0[:,2], system1[:,3]]
    actual_systems = verify_barlines(page, i, i+1, new_systems)
    if len(actual_systems):
        # Combine systems i and i+1
        page.systems[i] = dict(barlines=actual_systems,
                               start=page.systems[i]['start'],
                               stop=page.systems[i+1]['stop'])
        del page.systems[i+1]

def build_systems(page):
    initialize_systems(page)
    i = 0
    while i + 1 < len(page.systems): # on at most penultimate system
        if not try_join_system(page, i):
            i += 1

def show_system_barlines(page):
    import pylab as p
    for system in page.systems:
        for x0, x1, y0, y1 in system['barlines']:
            p.plot([x0, x1], [y0, y1], 'y')
