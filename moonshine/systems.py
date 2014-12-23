# Join adjacent staves into a system if a barline connects them.
from . import util, bitimage, hough
import numpy as np

def initialize_systems(page):
    """ Initial systems are each individual staff,
        barlines are a vertical line through the staff """
    page.systems = []
    for i, barlines in enumerate(page.barlines):
        system_bars = []
        for barline_range in barlines:
            barline_x = int(np.mean(barline_range))
            staff_y = page.staves.staff_y(i, barline_x)
            system_bars.append([[barline_x, staff_y - page.staff_dist*2],
                                [barline_x, staff_y + page.staff_dist*2]])
        barlines = np.array(system_bars, int)
        page.systems.append(dict(barlines=barlines, start=i, stop=i))

def verify_barlines(page, i, j, barlines):
    """ Convert each barline to a rho and theta value, then verify that
        a line exists using hough_lineseg. """
    # Get a slice of the image which includes systems i through j
    assert i <= j
    y0 = max(0, np.amin(page.staves()[i,:,1]) - page.staff_dist*3)
    y1 = min(page.img.shape[0],
             page.staves()[j,:,1].max() + page.staff_dist*3)
    # Gap between y0 and y1 must be a multiple of 8
    y1 += -(y1 - y0) & 7
    img_slice = page.img[y0:y1].copy()
    slice_T = bitimage.transpose(img_slice)
    slice_barlines = barlines.copy()
    slice_barlines[:,:,1] -= y0
    slice_barlines[slice_barlines < 0] = 0
    # Equations for transposed image:
    # rho = x cos t + y sin t
    # 0 = (x1 - x0) cos t + (y1 - y0) sin t
    # tan t = - (x1 - x0) / (y1 - y0)
    t = np.arctan(-(slice_barlines[:,1,0] - slice_barlines[:,0,0]).astype(float)
                   / (slice_barlines[:,1,1] - slice_barlines[:,0,1]))
    rho = slice_barlines[:,0,0] * np.cos(t) + slice_barlines[:,0,1] * np.sin(t)
    rhores = max(1, page.staff_dist / 4)
    rhoval = (rho / rhores).astype(int) + 1

    # Try a range of theta and rho values around each predicted line segment
    T_RANGE = np.linspace(-np.pi/100, np.pi/100, 11)
    RHO_RANGE = np.arange(-2, 3)
    dt = np.tile(T_RANGE, len(RHO_RANGE))
    drho = np.repeat(RHO_RANGE, len(T_RANGE))
    t = (t[:, None] + dt[None, :]).ravel()
    rhoval = (rhoval[:, None] + drho[None, :]).ravel()
    new_lines = hough.hough_lineseg_kernel(
                    slice_T, rhoval, t, rhores, max_gap=page.staff_dist).get()
    best_lines = hough.hough_paths(new_lines)
    best_lines = best_lines[:, :, ::-1] # transpose
    best_lines[:, :, 1] += y0
    best_lines = best_lines[(best_lines[:,0,1] < page.staves()[i,:,1].min()
                                                - page.staff_dist)
                          & (best_lines[:,1,1] > page.staves()[j,:,1].max()
                                                + page.staff_dist)]
    return best_lines
def try_join_system(page, i):
    """ Try joining system i with the system below it.
        Update page.systems, returning True,
        or return False if no barlines connect. """
    if (len(page.systems[i]['barlines']) == 0
        or len(page.systems[i+1]['barlines']) == 0):
        return False
    # Match each barline x1 in the top system to x0 in the bottom system
    # If no matching barline within staff_dist, then extend the barline
    # vertically to the height of the system
    system0 = page.systems[i]['barlines']
    system1 = page.systems[i+1]['barlines']
    system0_x1 = system0[:,1,0]
    system1_x0 = system1[:,0,0]
    pairs = util.merge(system0_x1, system1_x0, page.staff_dist)
    if len(pairs) == 0:
        return False
    ispair = (pairs >= 0).all(axis=1)
    if ispair.sum() == 0:
        # Must have at least one actual pair to test other possible ones
        return False
    s0_nopair = (pairs[:,0] >= 0) & ~ispair
    s1_nopair = (pairs[:,1] >= 0) & ~ispair
    new_systems = np.zeros((len(pairs), 2, 2), np.int32)
    new_systems[ispair, 0] = system0[pairs[ispair, 0], 0]
    new_systems[ispair, 1] = system1[pairs[ispair, 1], 1]
    new_systems[s0_nopair, 0] = system0[pairs[s0_nopair, 0], 0]
    new_systems[s0_nopair, 1] = system0[pairs[s0_nopair, 0], 1]
    new_systems[s0_nopair, 1, 1] = system1[:, 1, 1].mean()
    new_systems[s1_nopair, 1] = system1[pairs[s1_nopair, 1], 1]
    new_systems[s1_nopair, 0] = system1[pairs[s1_nopair, 1], 0]
    new_systems[s1_nopair, 0, 1] = system1[:, 0, 1].mean()
    actual_systems = verify_barlines(page,
                       page.systems[i]['start'],
                       page.systems[i+1]['stop'], new_systems)
    if len(actual_systems):
        # Combine systems i and i+1
        page.systems[i] = dict(barlines=actual_systems,
                               start=page.systems[i]['start'],
                               stop=page.systems[i+1]['stop'])
        del page.systems[i+1]
        return True
    else:
        return False

def build_systems(page):
    initialize_systems(page)
    i = 0
    while i + 1 < len(page.systems): # on at most penultimate system
        if not try_join_system(page, i):
            i += 1

def show_system_barlines(page):
    import pylab as p
    for system in page.systems:
        for (x0, y0), (x1, y1) in system['barlines']:
            p.plot([x0, x1], [y0, y1], 'y')
