import numpy as np

# (n, 2) ndarray of points in some staff's bounds
def get_points_in_staff(page):
  slice_heights = page.staff_line_positions[:, 4] \
                - page.staff_line_positions[:, 0]
  mask_ys = page.staff_line_positions[:, 0].repeat(slice_heights.ravel())
  # Add range of staff height to mask_ys
  ranges = [np.arange(l) for l in slice_heights.ravel()]
  mask_ys += np.concatenate(ranges)
  mask_xs = np.tile(np.arange(page.im.shape[1]),
                    page.staff_line_positions.shape[0]) \
              .repeat(slice_heights.ravel())
  masked = np.zeros(page.im.shape, dtype=bool)
  masked[mask_ys, mask_xs] = page.im[mask_ys, mask_xs]
  return np.column_stack(np.where(masked))

def vertical_hough(page, points):
  theta = np.linspace(-5.0*np.pi/180, 5.0*np.pi/180, 21)
  H = np.zeros((len(theta), page.im.shape[1]), dtype=int)
  for i, T in enumerate(theta):
    coeffs = np.array([[-np.sin(T)], [1]], dtype=np.double)
    coords = points.dot(coeffs)[:, 0]
    coords = np.rint(coords).astype(int)
    coords = coords[(coords >= 0) & (coords < page.im.shape[1])]
    bins = np.bincount(coords)
    H[i, 0:len(bins)] = bins
  for i in xrange(100):
    t, b = np.unravel_index(np.argmax(H), H.shape)
    print t, b
    if i == 0: extract_verticals(page, theta[t], b)
    H[max(0,t-1):t+2,max(0,b-2):b+3] = 0

def extract_verticals(page, t, b):
  ys = np.arange(page.im.shape[0])
  xs = b + np.rint(np.sin(t)*ys).astype(int)
  search_xs = np.tile(np.arange(-page.staff_thick, page.staff_thick + 1),
                      len(xs)) \
                .reshape((len(xs), 2*page.staff_thick + 1))
  search_xs += xs[:, np.newaxis]
  print ys.shape, search_xs.shape
  search_im = page.im[ys[:, np.newaxis], search_xs]
  candidate_rows = np.sum(search_im, axis=1) > 0
  # Detect number of swapped pixels which must be below staff_thick
  swaps = np.sum(np.bitwise_xor(search_im[:-1], search_im[1:]), axis=1)
  connected = (candidate_rows[:-1] & (swaps <= page.staff_thick)).astype(int)
  connected = np.concatenate((connected, [connected[-1]]))
  run_diff = np.diff(connected)
  run_starts, = np.where(run_diff == 1)
  run_ends, = np.where(run_diff == -1)
  run_inds = np.where((run_ends - run_starts) >= 4*page.staff_space)
  print zip(run_starts[run_inds], run_ends[run_inds])
class VerticalsTask:
  def __init__(self, page):
    self.page = page
  def process(self):
    points = get_points_in_staff(self.page)
    print vertical_hough(self.page, points)
  def color_image(self):
    pass
