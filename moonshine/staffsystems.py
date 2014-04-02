from .opencl import *
from . import hough
import numpy as np

# Detect staff systems by finding barlines that cross multiple staves
# The only barlines that remain in staff_filt cross multiple staves,
# and they start and end at the staff centers.
# Start at the first two staff centers and find vertical lines, then try
# to add more staves below until some of the lines don't span that far.
HOUGH_THETAS = np.linspace(-np.pi/500, np.pi/500, 11)
def build_staff_system(page, staff0):
    rhores = page.staff_thick
    # Round y0 down to nearest multiple of 8
    staff0min = min(page.staves[staff0, 2:4])
    y0 = staff0min & -8
    prev_measures = None
    for staff1 in xrange(staff0 + 1, len(page.staves)):
        # Round y1 up to nearest multiple of 8
        staff1max = max(page.staves[staff1, 2:4])
        y1 = -(-staff1max.astype(np.int32) & -8)
        print staff0min, staff1max, y0, y1
        img_slice = page.staff_filt[y0:y1].copy()
        # hough_line assumes almost horizontal lines so we need the transpose
        slice_T = bit_transpose_kernel(img_slice)
        slice_bins = hough_line_kernel(slice_T, rhores=rhores,
                                       numrho=slice_T.shape[0] // rhores,
                                       thetas=HOUGH_THETAS)
        max_bins = maximum_filter_kernel(slice_bins)
        measure_peaks = hough.houghpeaks(max_bins, npeaks=500)
        measure_theta = HOUGH_THETAS[measure_peaks[:, 0]]
        measure_rho = measure_peaks[:, 1]
        lines = hough_lineseg_kernel(slice_T, measure_rho, measure_theta,
                                     rhores=rhores,
                                     max_gap=page.staff_dist).get()
        

def system_measure_peaks(page):
    return build_staff_system(page, 0)
    #page.measure_bins = hough_line_kernel(img_T, rhores=rhores,
    #                                      numrho=img_T.shape[0] // rhores,
    #                                      thetas=HOUGH_THETAS)
    #max_bins = maximum_filter_kernel(page.measure_bins)
    #page.measure_peaks = hough.houghpeaks(max_bins, npeaks=100,
    #                                      invalidate=(11,
    #                                       page.staff_dist // page.staff_thick))
    return page.measure_peaks

def show_measure_peaks(page):
    import pylab as p
    for t, r in page.measure_peaks:
        # Draw transposed line
        theta = HOUGH_THETAS[t]
        rho = r * page.staff_thick
        p.plot([rho/np.cos(theta), (rho - 4096*np.sin(theta)) / np.cos(theta)],
             [0, 4096], 'y')
