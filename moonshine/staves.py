from .opencl import *
from . import hough
import numpy as np

def staff_center_lines(page):
    staff_filt = staffpoints_kernel(page.img, page.staff_dist)
    page.staff_filt = staff_filt
    thetas = np.linspace(-np.pi/500, np.pi/500, 51)
    rhores = page.staff_thick
    page.staff_bins = hough_line_kernel(staff_filt, rhores=rhores, numrho=page.img.shape[0] // rhores, thetas=thetas)
    # Some staves may have multiple Hough peaks so we need to take many more
    # peaks than the number of staves. Also, the strongest Hough response
    # doesn't always correspond to the longest segment, so we need many peaks
    # to find the longest segment, corresponding to the complete staff.
    # Most images shouldn't need this many peaks, but performance doesn't
    # seem to be an issue.
    peaks = hough.houghpeaks(page.staff_bins, npeaks=200)
    page.staff_peak_theta = thetas[peaks[:, 0]]
    page.staff_peak_rho = peaks[:, 1]
    lines = hough_lineseg_kernel(staff_filt, page.staff_peak_rho, page.staff_peak_theta, rhores=rhores, max_gap=page.staff_dist*4).get()
    page.staff_center_lines = lines
    return lines

def staves(page):
    lines = staff_center_lines(page)
    # Lines should all correspond to a center line of a staff.
    # Try to cluster with lines within 1 staff_dist and if a line could belong
    # to multiple clusters, something has gone horribly wrong.
    # Each cluster should then be one staff, so assign the longest segment
    # from each cluster as a staff.

    # Map each y position in the image to the index of the cluster
    # -1: not assigned yet
    cluster_id = -np.ones(page.img.shape[0], int)
    clusters = []

    for x0, x1, y0, y1 in lines:
        ymin = min(y0, y1)
        ymax = max(y0, y1)
        line_cluster = cluster_id[ymin:ymax+1]
        if (line_cluster == -1).all():
            # Assign a new cluster
            cluster_num = len(clusters)
            cluster_id[ymin:ymax+1] = cluster_num
            clusters.append([(x0, x1, y0, y1)])
        else:
            cluster_num = line_cluster[line_cluster >= 0][0]
            # The cluster the line belongs to should not be ambiguous
            if ((line_cluster != -1) & (line_cluster != cluster_num)).any():
                raise Error("Error detecting staves")
            
        clusters[cluster_num].append((x0, x1, y0, y1))
        cluster_id[max(0, ymin - page.staff_dist*4)
                   : min(page.img.shape[0], ymax + page.staff_dist*4)] = cluster_num

    # Take the longest segment from each cluster
    staves = []
    for lines in clusters:
        lines = np.array(lines)
        staves.append(lines[np.argmax(lines[:, 1] - lines[:, 0])])
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
