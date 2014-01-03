import numpy as np

def bincount_2d(samples):
  """ Bin samples which are in rows with 2 columns of observations. """
  samples = samples.astype(int)
  stride = np.amax(samples[:, 1]) + 1
  vals = samples[:, 0] * stride + samples[:, 1]
  bins = np.bincount(vals)
  rows = -(-len(bins) / stride)
  cols = stride
  bins.resize(rows * cols, refcheck=False)
  return bins.reshape((rows, cols))

def hough_line(image, ts):
  """ Detect line in image with intercept on the 0th axis and angle to the
      horizontal t (in ts). Return accumulators with intercept on the
      0th axis and t on the 1st axis.
  """
  # Stack coordinates of dark pixels in image
  coords = np.c_[np.where(image)]
  # Multiply coords . coeffs to get intercepts for each t value for each point
  # y0 = y - tan(t)*x
  coeffs = np.vstack([np.ones(len(ts)),
                      -np.tan(ts)])
  intercepts = np.rint(coords.dot(coeffs)).astype(int)
  # Extract original point and t value belonging to intercepts
  in_range = (0 <= intercepts) & (intercepts < image.shape[0])
  intercept_point, intercept_t = np.where(in_range)
  # 1D intercepts which are in range
  intercept_val = intercepts[intercept_point, intercept_t]
  return bincount_2d(np.c_[intercept_val, intercept_t])

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
  bin_max = scipy.ndimage.maximum_filter(bins, size=(bins.shape[0]/10, bins.shape[1]/10))
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
