from .opencl import *
from . import hough
import numpy as np
import scipy.cluster.hierarchy

def staff_center_lines(page):
    staff_filt = staffpoints_kernel(page.img, page.staff_dist)
    page.staff_filt = staff_filt
    thetas = np.linspace(-np.pi/500, np.pi/500, 51)
    rhores = page.staff_thick*2
    page.staff_bins = hough_line_kernel(staff_filt, rhores=rhores, numrho=page.img.shape[0] // rhores, thetas=thetas)
    # Some staves may have multiple Hough peaks so we need to take many more
    # peaks than the number of staves. Also, the strongest Hough response
    # doesn't always correspond to the longest segment, so we need many peaks
    # to find the longest segment, corresponding to the complete staff.
    # Most images shouldn't need this many peaks, but performance doesn't
    # seem to be an issue.
    peaks = hough.houghpeaks(page.staff_bins, thresh=max_kernel(page.staff_bins)/4.0)
    page.staff_peak_theta = thetas[peaks[:, 0]]
    page.staff_peak_rho = peaks[:, 1]
    lines = hough_lineseg_kernel(staff_filt,
                                 page.staff_peak_rho, page.staff_peak_theta,
                                 rhores=rhores, max_gap=page.staff_dist*2).get()
    page.staff_center_lines = lines
    return lines

def staves(page):
    lines = staff_center_lines(page)
    # Lines should all correspond to a center line of a staff.
    # Try to cluster with lines within 1 staff_dist and if a line could belong
    # to multiple clusters, something has gone horribly wrong.
    # Each cluster should then be one staff, so assign the longest segment
    # from each cluster as a staff.

    # Staff center line ys should be separated by a clear margin
    # Using SciPy's hierarchical clustering for this is overkill
    staff_ids = scipy.cluster.hierarchy.fclusterdata(
                    np.mean(lines[:, 2:4], 1)[:, None],
                    page.staff_dist,
                    criterion="distance", method="complete")
    num_staves = np.amax(staff_ids)
    staves = []
    for s in xrange(1, num_staves + 1):
        staff_candidates = lines[staff_ids == s]
        staff_width = staff_candidates[:, 1] - staff_candidates[:, 0]
        staff = staff_candidates[np.argmax(staff_width)]
        if staff[1] - staff[0] > page.img.shape[1] * 8 / 2:
            staves.append(staff)
    staves = np.array(staves)
    staves = staves[np.argsort(staves[:, 2])]
    page.staves = staves
    return staves

def show_staff_centers(page):
    import pylab as p
    # Overlay staff line points
    staff_filt = np.unpackbits(page.staff_filt.get()).reshape((4096, -1))
    staff_line_mask = np.ma.masked_where(staff_filt == 0, staff_filt)
    p.imshow(staff_line_mask, cmap='Greens')
    for x0, x1, y0, y1 in page.staves:
        p.plot([x0, x1], [y0, y1], 'g')
