from .. import hough
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
                 ts=np.linspace(-self.HOUGH_LORES_THETA*self.HOUGH_NUM_THETA,
                                 self.HOUGH_LORES_THETA*self.HOUGH_NUM_THETA,
                                2*self.HOUGH_NUM_THETA+1))

  def crop_staff_peak(self, r, t):
    """ r, t: index into self.H
        the line should be almost horizontal (t ~= 0)
        return cropped image around possible staff area, image bounds,
        and adjusted real-valued r and t
    """
    theta = (t - self.HOUGH_NUM_THETA)*self.HOUGH_LORES_THETA
    rho0 = np.double(r*self.page.staff_thick)
    b0 = rho0 / np.cos(theta) # y-intercept
    if np.cos(theta) >= 0:
      # y = (rho - x sin theta) / cos theta
      ymin = rho0 / np.cos(theta)
      ymax = (rho0 - self.page.im.shape[1] * np.sin(theta)) / np.cos(theta)
    ymin = int(np.rint(ymin)) - self.page.staff_dist*5
    ymax = int(np.rint(ymax)) + self.page.staff_dist*5 + 1
    img = self.page.im[ymin:ymax]
    rho = rho0 * (b0 - ymin) / b0
    return img, slice(ymin,ymax), rho, theta
    
  def staff_from_peak(self, r, t):
    theta0 = (t - self.HOUGH_NUM_THETA) * self.HOUGH_LORES_THETA
    # Build hi-res Hough transform centered on (r, t) and calculate
    # the corresponding peak

def normal_to_slope(r, t):
    return (-1.0/np.tan(t), r/np.sin(t))

def hough_line_segment(img, rho_res, rho, theta, max_gap=0):
    """ Return single longest line segment in Hough bin """
    # Select pixels which are in the path of the line
    hough_pixels = np.zeros_like(img, dtype=np.uint8)
    # Draw top boundary
    r0 = np.double(rho * rho_res) - (rho_res / 2.0)
    xs = np.arange(hough_pixels.shape[1])
    m, b = normal_to_slope(r0, np.pi/2 - theta)
    ys = np.rint(m*xs + b).astype(int)
    first_pixel = np.amin(ys[[0, -1]])
    in_range = (0 <= ys) & (ys < hough_pixels.shape[0])
    hough_pixels[ys[in_range], xs[in_range]] = 1
    # Draw bottom boundary
    m, b = normal_to_slope(r0 + rho_res, np.pi/2 - theta)
    ys = np.rint(m*xs + b).astype(int)
    last_pixel = np.amax(ys[[0, -1]])
    in_range = (0 <= ys) & (ys < hough_pixels.shape[0])
    hough_pixels[ys[in_range], xs[in_range]] += 1

    # Narrow img and hough_pixels to vertical interval [first_pixel, last_pixel]
    hough_pixels = hough_pixels[first_pixel : last_pixel+1]
    img = img[first_pixel : last_pixel+1]
    np.cumsum(hough_pixels, axis=0, out=hough_pixels)
    hough_pixels[hough_pixels != 1] = 0
    #ypix, xpix = np.where(img & hough_pixels.astype(bool))
    xpix, = np.where((img & hough_pixels.astype(bool)).any(axis=0))
    # XXX: should rotate ypix and xpix values if line is not horizontal
    x_segments = np.zeros(img.shape[1], bool)
    x_bins = np.bincount(xpix) # number pixels in each vertical slice
    x_segments[:len(x_bins)] = x_bins > 0
    line_x, = np.where(x_segments)
    gap_len = np.diff(line_x)
    gap_fill = (1 < gap_len) & (gap_len <= max_gap)
    gap_start = line_x[:-1]
    gap_end = line_x[1:]
    for gap_ind in np.where(gap_fill)[0]:
        x_segments[gap_start[gap_ind]:gap_end[gap_ind]] = True
    # Return maximum continuous range in x_segments
    x_seg_start = x_segments == True
    x_seg_start[1:] &= x_segments[:-1] == False
    x_seg_end = x_segments == False
    x_seg_end[0] = False
    x_seg_end[1:] &= x_segments[:-1] == True
    x_seg_start, = np.where(x_seg_start)
    x_seg_end, = np.where(x_seg_end)
    segment_ind = np.argmax(x_seg_end - x_seg_start)
    return [x_seg_start[segment_ind], x_seg_end[segment_ind]]

def hough_extract_staff(page, Hcol, rho_res, rho, theta):
    staff_rho = []
    staff_min = []
    staff_max = []
    gap = page.staff_dist*8
    rhomin = rhomax = rho
    while True:
        xmin, xmax = hough_line_segment(page.im != 0, rho_res, rho, theta, max_gap=gap)
        # Criteria for stopping
        if xmin > page.im.shape[1] / 4 or xmax < page.im.shape[1] * 3 / 4:
            print 'break', rho, (xmin,xmax), 'from page size' , page.im.shape[1]
            break
        elif staff_min:
            if (np.abs(xmin - np.mean(staff_min)) > gap
                or np.abs(xmax - np.mean(staff_max)) > gap):
                print 'break', rho, (xmin,xmax), 'from', staff_min, staff_max
                break
        staff_rho.append(rho)
        staff_min.append(xmin)
        staff_max.append(xmax)
        rho0_above = rhomin - page.staff_dist / page.staff_thick - 1
        rho_above = rho0_above + Hcol[rho0_above:rho0_above+3].argmax()
        rho0_below = rhomax + page.staff_dist / page.staff_thick - 1
        rho_below = rho0_below + Hcol[rho0_below:rho0_below+3].argmax()
        if Hcol[rho_above] > Hcol[rho_below]:
            rhomin = rho = rho_above
        else:
            rhomax = rho = rho_below
    return staff_rho

if __name__=="__main__":
    for i in xrange(13):
        st = sorted(hough_extract_staff(page, H[:,t], page.staff_thick, r, (t-10)*np.pi/1000))
        print st
        if len(st):
          H[st[0]-2:st[-1]+2, :] = 0
        else:
          H[r-2:r+2, :] = 0
        r,t = np.unravel_index(H.argmax(), H.shape)
