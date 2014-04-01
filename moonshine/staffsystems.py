from .opencl import *
from . import hough
import numpy as np

# Detect system measures (barlines which cross multiple staves)
HOUGH_THETAS = np.linspace(-np.pi/100, np.pi/100, 51)
def system_measure_peaks(page):
    img_T = bit_transpose_kernel(page.img)
    rhores = page.staff_thick
    page.measure_bins = hough_line_kernel(img_T, rhores=rhores,
                                          numrho=img_T.shape[0] // rhores,
                                          thetas=HOUGH_THETAS)
    page.measure_peaks = hough.houghpeaks(page.measure_bins, npeaks=50)
    return page.measure_peaks

def show_measure_peaks(page):
    import pylab as p
    for t, r in page.measure_peaks:
        # Draw transposed line
        theta = HOUGH_THETAS[t]
        rho = r * page.staff_thick
        p.plot([rho/np.cos(theta), (rho - 4096*np.sin(theta)) / np.cos(theta)],
             [0, 4096], 'y')
