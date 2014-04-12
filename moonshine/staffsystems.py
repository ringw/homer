from .opencl import *
from . import hough, bitimage, filter
import numpy as np
import logging
logger = logging.getLogger('systems')

# Detect staff systems by finding barlines that cross multiple staves
# The only barlines that remain in staff_filt cross multiple staves,
# and they start and end at the staff centers.
# Start at the first two staff centers and find vertical lines, then try
# to add more staves below until some of the lines don't span that far.
HOUGH_THETAS = np.linspace(-np.pi/100, np.pi/100, 11)
def build_staff_system(page, staff0):
    barlines = np.zeros((0, 4))
    if staff0 + 1 == len(page.staves):
        return (staff0, staff0 + 1, barlines)
    rhores = page.staff_thick
    # Round y0 down to nearest multiple of 8
    staff0min = min(page.staves[staff0, 2:4])
    staff0min = max(0, staff0min - page.staff_dist * 2)
    y0 = max(0, staff0min - page.staff_dist) & -8
    prev_measures = None
    staff1 = staff0 + 1 # start with single staff
    for staff1 in xrange(staff0 + 1, len(page.staves)):
        # Round y1 up to nearest multiple of 8
        staff1max = max(page.staves[staff1, 2:4])
        staff1max = min(page.img.shape[0], staff1max + page.staff_dist * 2)
        y1 = -(-min(page.img.shape[0], staff1max + page.staff_dist)
                    .astype(np.int32) & -8)
        img_slice = page.barline_filt[y0:y1].copy()
        # hough_line assumes almost horizontal lines so we need the transpose
        slice_T = bitimage.transpose(img_slice)
        slice_bins = hough_line_kernel(slice_T, rhores=rhores,
                                       numrho=slice_T.shape[0] // rhores,
                                       thetas=HOUGH_THETAS)
        max_bins = maximum_filter_kernel(slice_bins)
        measure_peaks = hough.houghpeaks(max_bins,
                                         thresh=max_kernel(max_bins)/8.0)
        measure_theta = HOUGH_THETAS[measure_peaks[:, 0]]
        measure_rho = measure_peaks[:, 1]
        lines = hough_lineseg_kernel(slice_T, measure_rho, measure_theta,
                                     rhores=rhores,
                                     max_gap=page.staff_dist).get()
        is_barline = ((np.abs(lines[:, 0].astype(int) - (staff0min - y0))
                            < page.staff_dist // 2)
                      & (np.abs(lines[:, 1].astype(int) - (staff1max - y0))
                            < page.staff_dist // 2))
        new_barlines = lines[is_barline]
        if len(new_barlines):
            actual_barlines = hough.hough_paths(new_barlines)
            if len(actual_barlines) == 0:
                break
            else:
                barlines = (actual_barlines[:, [2, 3, 0, 1]]
                                + [0, 0, y0, y0])

            staff1 += 1
        else:
            # Previously we only had one staff or else we had a staff system
            # with some barline
            break
    return (staff0, staff1, barlines)

def staff_systems(page):
    staff0 = 0
    page.barline_filt = filter.barline_filter(page)
    page.barlines = []
    while staff0 < len(page.staves):
        staff0, staff1, barlines = build_staff_system(page, staff0)
        page.barlines.append((staff0, staff1, barlines))
        staff0 = staff1
    logger.debug("Systems contain "
                 + ",".join([str(s1-s0) for (s0,s1,barlines) in page.barlines])
                 + " staves")
    return page.barlines

def show_measure_peaks(page):
    import pylab as p
    for t, r in page.staff_systems:
        # Draw transposed line
        theta = HOUGH_THETAS[t]
        rho = r * page.staff_thick
        p.plot([rho/np.cos(theta), (rho - 4096*np.sin(theta)) / np.cos(theta)],
             [0, 4096], 'y')

def show_barlines(page):
    import pylab as p
    for start, end, barlines in page.barlines:
        for x0, x1, y0, y1 in barlines:
            p.plot([x0, x1], [y0, y1], 'y')
