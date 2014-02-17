import ..hough
import numpy as np
from scipy.ndimage import maximum_filter

class StaffLines:
  HOUGH_LORES_THETA = np.pi / 500
  HOUGH_HIRES_THETA = np.pi / 2500
  HOUGH_NUM_THETA   = 5
  def __init__(self, page):
    self.page = page
    # Hough transform with low theta resolution
    # (we will later build a Hough transform with a higher resolution
    # centered on each peak in this image)
    self.H = hough.hough_line(self.page.im != 0,
                              rho_res=self.page.staff_thick,
                              ts=np.linspace(-HOUGH_LORES_THETA*HOUGH_NUM_THETA,
                                              HOUGH_LORES_THETA*HOUGH_NUM_THETA,
                                              2*HOUGH_NUM_THETA+1))

  def crop_staff_peak(self, r, t):
    """ r, t: index into self.H
        the line should be almost horizontal (t ~= 0)
        return cropped image around possible staff area, image bounds,
        and adjusted real-valued r and t
    """
    
  def staff_from_peak(self, r, t):
    theta0 = (t - HOUGH_NUM_THETA) * HOUGH_LORES_THETA
    # Build hi-res Hough transform centered on (r, t) and calculate
    # the corresponding peak
