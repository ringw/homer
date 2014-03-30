from .opencl import *
from . import hough
import numpy as np

def staffpoints(page):
    staff_filt = staffpoints_kernel(page.img, page.staff_dist)
    page.staff_filt = staff_filt
    thetas = np.linspace(-np.pi/100, np.pi/100, 25)
    page.staff_bins = hough_line_kernel(staff_filt, rhores=1, numrho=page.img.shape[0], thetas=thetas)
    peaks = hough.houghpeaks(page.staff_bins, npeaks=50)
    page.staff_peak_theta = thetas[peaks[:, 0]]
    page.staff_peak_rho = peaks[:, 1]
    segments = hough_lineseg_kernel(staff_filt, page.staff_peak_rho, page.staff_peak_theta).get()
    print segments[np.argsort(segments[:,2])]

def show_stafflines(page):
    from pylab import *
    # Overlay staff line points
    staff_filt = np.unpackbits(page.staff_filt.get()).reshape((4096, -1))
    staff_line_mask = np.ma.masked_where(staff_filt == 0, staff_filt)
    imshow(staff_line_mask, cmap='Greens')
    for t, r in zip(page.staff_peak_theta, page.staff_peak_rho):
        x0 = 0
        x1 = 4096
        y0 = r / np.cos(t)
        y1 = (r - x1 * np.sin(t)) / np.cos(t)
        plot([x0, x1], [y0, y1], 'g')
