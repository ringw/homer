import numpy as np
import ImageDraw

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
  theta = np.linspace(-2.0*np.pi/180, 2.0*np.pi/180, 21)
  H = np.zeros((len(theta), page.im.shape[1]), dtype=int)
  for i, T in enumerate(theta):
    coeffs = np.array([[-np.sin(T)], [1]], dtype=np.double)
    coords = points.dot(coeffs)[:, 0]
    coords = np.rint(coords).astype(int)
    coords = coords[(coords >= 0) & (coords < page.im.shape[1])]
    bins = np.bincount(coords)
    H[i, 0:len(bins)] = bins
  # Array of ndarrays to be concatenated
  verticals = []
  SCORE_MIN = page.staff_thick * page.staff_space * 5
  while np.count_nonzero(H):
    t, b = np.unravel_index(np.argmax(H), H.shape)
    runs, score = extract_verticals(page, theta[t], b)
    if score < SCORE_MIN:
      break
    # y0, y1, x0, x1
    V = np.zeros((runs.shape[0], 4), dtype=np.double)
    V[:, 0:2] = runs
    V[:, 2:4] = b
    V[:, 2:4] += V[:, 0:2] * np.sin(theta[t])
    verticals.append(V)
    ts = slice(max(0,t-3),min(21,t+4))
    bs = slice(max(0,b-page.staff_space),min(page.im.shape[1], b+page.staff_space+1))
    H[ts,bs] = 0
  return np.concatenate(verticals)

def extract_verticals(page, t, b):
  ys = np.arange(page.im.shape[0])
  xs = b + np.rint(np.sin(t)*ys).astype(int)
  SEARCH_WIDTH = max(1, page.staff_thick/2)
  search_xs = np.tile(np.arange(-SEARCH_WIDTH, SEARCH_WIDTH + 1),
                      len(xs)) \
                .reshape((len(xs), 2*SEARCH_WIDTH + 1))
  search_xs += xs[:, np.newaxis]
  # XXX: Don't consider lines which go off edge of image
  if np.count_nonzero((search_xs < 0) | (search_xs >= page.im.shape[1])):
    return np.zeros((0, 2), dtype=int), np.iinfo(int).max
  #print ys.shape, search_xs.shape
  search_im = page.im[ys[:, np.newaxis], search_xs]
  score = np.sum(search_im)
  candidate_rows = np.sum(search_im, axis=1) > 0
  # Detect number of toggled pixels which must be below TOGGLE_THRESHOLD
  TOGGLE_THRESHOLD = max(1, page.staff_thick/2)
  toggles = np.sum(np.bitwise_xor(search_im[:-1], search_im[1:]), axis=1)
  connected = (candidate_rows[:-1] & (toggles <= TOGGLE_THRESHOLD)).astype(int)
  connected = np.concatenate((connected, [connected[-1]]))
  # Skip over gaps less than staff_thick
  can_skip = np.convolve(connected,
                         np.concatenate(([1], [0 for i in xrange(page.staff_thick)], [1])),
                         mode='same')
  connected += (can_skip == 2)
  connected[connected > 1] = 1
  run_diff = np.diff(connected)
  run_starts, = np.where(run_diff == 1)
  run_ends, = np.where(run_diff == -1)
  run_ends += 1
  run_inds = np.where((run_ends - run_starts) >= 4*page.staff_space)
  return np.column_stack((run_starts[run_inds], run_ends[run_inds])), score

def extract_barlines(page, verticals):
  # Pairwise distance from top of each vertical to top of each staff
  DISTS_SHAPE = (len(verticals), len(page.staff_line_positions))
  staff_nums = np.tile(np.arange(len(page.staff_line_positions)),
                       len(verticals)).reshape(DISTS_SHAPE)
  vertical_xs = np.rint(verticals[:, 2]) \
                  .repeat(len(page.staff_line_positions)) \
                  .reshape(DISTS_SHAPE).astype(int)
  staff_top_dists = page.staff_line_positions[staff_nums, 0, vertical_xs]
  staff_top_dists -= verticals[:, 0].repeat(len(page.staff_line_positions)) \
                                    .reshape(DISTS_SHAPE)
  staff_top_dists = np.abs(staff_top_dists)
  staff_bot_dists = page.staff_line_positions[staff_nums, -1, vertical_xs]
  staff_bot_dists -= verticals[:, 1].repeat(len(page.staff_line_positions)) \
                                    .reshape(DISTS_SHAPE)
  staff_bot_dists = np.abs(staff_bot_dists)
  #np.set_printoptions(threshold=np.nan)
  top_staves = np.argmin(staff_top_dists, axis=1)
  bot_staves = np.argmin(staff_bot_dists, axis=1)
  top_dists = staff_top_dists[np.arange(len(staff_top_dists)), top_staves]
  bot_dists = staff_bot_dists[np.arange(len(staff_bot_dists)), bot_staves]
  MAX_DIST = 2*page.staff_thick
  barlines, = np.where((top_dists <= MAX_DIST) & (bot_dists <= MAX_DIST))
  x = np.column_stack((top_staves,bot_staves,top_dists,bot_dists))
  return verticals[barlines]

class VerticalsTask:
  def __init__(self, page):
    self.page = page
  def process(self):
    if len(self.page.staff_line_positions) == 0: return False
    points = get_points_in_staff(self.page)
    self.verticals = vertical_hough(self.page, points)
    self.page.barlines = extract_barlines(self.page, self.verticals)
  def color_image(self):
    d = ImageDraw.Draw(self.page.colored)
    for barline in self.page.barlines:
      d.line(tuple(np.rint(barline[[2, 0, 3, 1]]).astype(int)), fill=(0,0,255))
