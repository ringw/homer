import numpy as np

def bincount_2d(samples, shape=None):
  """ Bin samples which are in rows with 2 columns of observations. """
  samples = samples.astype(int)
  stride = np.amax(samples[:, 1]) + 1
  vals = samples[:, 0] * stride + samples[:, 1]
  bins = np.bincount(vals)
  rows = -(-len(bins) / stride)
  cols = stride
  bins.resize(rows * cols, refcheck=False)
  bins = bins.reshape((rows, cols))
  if shape is None or shape == bins.shape:
    return bins
  else:
    new_bins = np.zeros(shape, np.int)
    new_bins[:rows, :cols] = bins
    return new_bins

def hough_line(image, ts=np.linspace(0, 2*np.pi, 60), rho_res=1, bins=None):
  """ Detect line in image with intercept on the 0th axis and angle to the
      horizontal t (in ts). Return accumulators with intercept on the
      0th axis and t on the 1st axis.
  """
  num_rho = int(np.ceil(np.sqrt(image.shape[0]**2 + image.shape[1]**2) / rho_res))
  # Stack coordinates of dark pixels in image
  coords = np.c_[np.where(image)]
  # Multiply coords . coeffs to get rho for each t value for each point
  # rho = x cos t + y sin t
  coeffs = np.r_[[np.cos(ts),
                  np.sin(ts)]]
  rho_vals = coords.dot(coeffs)
  rho_index = np.rint(rho_vals / rho_res).astype(int)
  in_range = (0 <= rho_index) & (rho_index < num_rho)
  rho_index[~ in_range] = num_rho
  # We want to bincount each column (t value), but the results of np.bincount
  # may be a different length. We need to add an additional row with fake
  # out-of-range rho values (set to num_rho) so that all columns are length
  # num_rho + 1. Then we discard the last row which counts all out-of-range
  # rho values.
  rho_index = np.vstack([rho_index, np.repeat(num_rho, len(ts))])
  H = np.apply_along_axis(np.bincount, 0, rho_index)
  return H[:-1]

def hough_peaks(bins, size=None, min_val=1):
  if size is None:
    size = tuple(s/10 for s in bins.shape)
  ys = []
  xs = []
  bins = bins.copy()
  while (bins >= min_val).any():
    y, x = np.unravel_index(bins.argmax(), bins.shape)
    ys.append(y)
    xs.append(x)
    ymin = max(0, y - size[0]/2)
    ymax = max(bins.shape[0], y + size[0]/2)
    ymax = min(bins.shape[0], y -(-size[0]/2))
    xmin = max(0, x - size[1]/2)
    xmax = min(bins.shape[1], x -(-size[1]/2))
    bins[ymin:ymax, xmin:xmax] = 0
  return np.c_[ys, xs]

def hough_line_plot(bins, ts, image_shape, min_val=1):
  import pylab
  import scipy.ndimage
  bin_max = scipy.ndimage.maximum_filter(bins, size=(bins.shape[0]/10,
                                                     bins.shape[1]/10))
  lines = (bins == bin_max) & (bins >= min_val)
  unique_lines, num_lines = scipy.ndimage.label(lines)
  for i in xrange(1, num_lines+1):
    intercept, t = np.where(unique_lines == i)
    intercept = np.mean(intercept)
    t = np.mean(t)
    slope = np.tan(ts[t])
    pylab.plot([0, image_shape[1]],
               [intercept, intercept + slope*image_shape[1]], 'g')
  pylab.xlim([0, image_shape[1]])
  pylab.ylim([0, image_shape[0]])
