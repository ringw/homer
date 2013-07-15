import numpy as np
import Image
import ImageDraw

class Staff:
  def __init__(self):
    self.lines = tuple([] for i in range(5))
    self.line_masks = list(None for i in range(5))
  def add_point(self, x, ys):
    # Insert points in ys into each line at position x
    for i in range(5):
      y = ys[i]
      line = self.lines[i]
      line_ind = 0
      line_len = len(line)
      while line_ind < line_len:
        if line[line_ind][0] >= x:
          break
        line_ind += 1
      line.insert(line_ind, (x, y))
  def draw(self, im):
    d = ImageDraw.Draw(im)
    for line in self.lines:
      d.line(line, fill=(255, 255, 0))
class StaffTask:
  def __init__(self, page):
    self.page = page
    self.im = page.im
    self.colored = page.colored

  STAFF_SPACE_DY = 10
  STAFF_THICK_DY = 5

  def get_spacing(self):
    dark_cols = np.array(self.page.col_runs[:,4], dtype=bool)

    # Histogram of light lengths (space between staff lines)
    dists = np.bincount(self.page.col_runs[~dark_cols, 3])
    # Histogram of dark lengths (thickness of staff lines)
    thicks = np.bincount(self.page.col_runs[dark_cols, 3])

    self.page.staff_space = self.staff_space = np.argmax(dists)
    self.page.staff_thick = self.staff_thick = np.argmax(thicks)
    return (self.staff_space, self.staff_thick)

  # 1D array -> 5-tuples of coordinates of possible staff cross-sections
  def cross_sections(self, col_num):
    # Extract runs from this column
    runs = self.page.col_runs[self.page.col_runs[:, 0] == col_num]
    if runs.shape[0] < 9: return []
    # Ensure first run is dark
    if not runs[0,4]:
      runs = runs[1:]
    
    # Runs alternate color, so there are (len(runs) + 1)/2 - 4 candidates
    num_candidates = (runs.shape[0] + 1)/2 - 4
    candidate_starts = np.arange(0, 2*num_candidates, 2)
    # Consecutive 9-tuples of candidate staff cross-sections with run lengths
    # (dark, light, dark, ..., dark)
    sections = np.zeros((num_candidates, 9), dtype=int)
    for i in xrange(9):
      sections[:,i] = runs[candidate_starts + i, 3]
    thicks = sections[:,[0,2,4,6,8]]
    dists = sections[:,[1,3,5,7]]

    # Find indices of staff cross-sections
    candidates = (np.abs(np.mean(thicks, axis=1) - self.staff_thick) \
                    < self.STAFF_THICK_DY) \
                 & (np.abs(np.mean(dists, axis=1) - self.staff_space) \
                       < self.STAFF_SPACE_DY) \
                 & (np.std(thicks, axis=1) < 1.0) \
                 & (np.std(dists, axis=1) < 3.0)
    candidate_ind = np.arange(0, runs.shape[0])[candidates]
    # Find staff line center y from runs position and length
    centers = np.zeros((np.count_nonzero(candidates), 5))
    for i in xrange(5):
      centers[:,i] = np.sum(runs[(candidate_ind + i)*2, 1:3], axis=1)/2

    return centers

  # Find best cross-section for searching for staves in each horizontal interval
  NUM_INTERVALS = 100
  def search_staff_intervals(self):
    # Generate evenly spaced, equally sized intervals
    step = int(np.ceil(float(self.im.shape[1]) / self.NUM_INTERVALS))
    boundaries = np.linspace(0, self.im.shape[1], num=20)
    intervals = np.vstack((boundaries[:-1], boundaries[1:]))
    # Sum columns for vertical projection to choose cross section
    proj = self.im.sum(axis=0)
    proj.resize(self.NUM_INTERVALS * step)
    proj.shape = (self.NUM_INTERVALS, step)
    PROJECTION_MIN = self.im.shape[0] / 100
    # Replace projections less than PROJECTION_MIN to INT_MAX to mark them bad
    INT_MAX = np.iinfo(proj.dtype).max
    proj[proj < PROJECTION_MIN] = INT_MAX
    # Choose cross-section from minimum in each interval
    section_nums = np.argmax(proj, axis=1)
    # Search chosen cross-sections
    for section, ind in enumerate(section_nums):
      if proj[section, ind] == INT_MAX:
        continue
      i = section*step + ind
      new_sections = self.cross_sections(i)
      num_old_sections = self.sections.shape[0]
      self.sections.resize((num_old_sections + new_sections.shape[0], 6))
      self.sections[num_old_sections:, 0] = i
      self.sections[num_old_sections:, 1:] = new_sections

  # Hough transform using center points of staff cross-sections
  HOUGH_DEGREES_MAX = 10.0
  HOUGH_THETA_RESOLUTION = 0.1
  NUM_DEGREES = np.ceil(2*HOUGH_DEGREES_MAX/HOUGH_THETA_RESOLUTION + 1)
  def build_staves(self):
    theta = np.linspace(-self.HOUGH_DEGREES_MAX, self.HOUGH_DEGREES_MAX,
                        self.NUM_DEGREES)
    coords = np.zeros((self.sections.shape[0], 2), dtype=np.double)
    coords[:, 0] = self.sections[:, 0] # abscissa of cross-sections
    coords[:, 1] = np.mean(self.sections[:, 1:], axis=1) # mean of staff lines
    # Matrix multiplied with coords to get rho values to be incremented
    coeffs = np.vstack((-np.sin(theta*np.pi/180.0), np.ones(len(theta))))
    rhos = coords.dot(coeffs)
    # Create histogram from flattened rho and theta values
    rho_vals = rhos.ravel()
    rho_ind, theta_ind = np.unravel_index(np.arange(0, len(rho_vals)),
                                          rhos.shape)
    # Histogram bins contain each y-value and theta value
    H = np.histogram2d(rhos.ravel(), theta[theta_ind],
                        bins=[self.im.shape[0], self.NUM_DEGREES],
                        range=[[-0.5, self.im.shape[0] + 0.5],
                               [theta[0] - self.HOUGH_THETA_RESOLUTION/2,
                                theta[-1] + self.HOUGH_THETA_RESOLUTION/2]])[0]
    # Remove best candidates until one does not represent a viable staff
    # Don't add a staff if it shares any sections with an already added staff
    retrieved_sections = np.zeros(self.sections.shape[0], dtype=bool)
    clusters = []
    self.staves = []
    # We expect much less than 30 staves on a page;
    # allow for some staves to have less cross sections
    expected_sections = self.sections.shape[0] / 30
    while np.count_nonzero(retrieved_sections == 0):
      y, t = np.unravel_index(np.argmax(H), H.shape)
      if H[y, t] == 0: break
      # Multiply with sections to get top, center, bottom y-intercept
      coeffs = np.array([[-np.sin(theta[t]*np.pi/180) for i in (1,2,3)],
                         [1, 0, 0],
                         [0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0],
                         [0, 0, 1]], dtype=np.double)
      section_intercepts = self.sections.dot(coeffs)
      intersects = (section_intercepts[:, 0] <= y) & (y <= section_intercepts[:, 2])
      if np.count_nonzero(intersects) < expected_sections:
        break
      elif not (retrieved_sections & intersects).any():
        retrieved_sections |= intersects
        staff = Staff()
        # Choose staff sections where center line is closest to our line
        xs = np.unique(self.sections[:, 0])
        for x in xs:
          distance = np.abs(section_intercepts[:, 1] - y)
          candidates = intersects & (self.sections[:, 0] == x) \
                       & (distance < self.staff_space/2)
          if np.count_nonzero(candidates):
            sections = self.sections[candidates]
            distances = distance[candidates]
            staff.add_point(x, sections[np.argmin(distances), 1:])
        self.staves.append(staff)
        # Invalidate anything that would (probably) intersect with this staff
        H[y-self.staff_space*5:y+self.staff_space*5, t-10:t+10] = 0
      else:
        # Invalidate very close points
        H[y-self.staff_space:y+self.staff_space, t] = 0

    # Sort staves by start of top line
    staff_tops = np.array([staff.lines[0][0][1] for staff in self.staves])
    self.staves = list(np.array(self.staves)[np.argsort(staff_tops)])

  def interpolate_staff_line_positions(self):
    self.line_positions = np.zeros((len(self.staves), 5, self.im.shape[1]),
                                   dtype=int)
    for i, staff in enumerate(self.staves):
      sections = np.hstack(staff.lines)
      section_xs = sections[:, 0]
      section_ys = sections[:, 1::2]

      x_pairs = np.zeros((len(section_xs) + 1, 2))
      x_pairs[1:, 0] = x_pairs[:-1, 1] = section_xs
      x_pairs[-1, 1] = self.im.shape[1]
      y_pairs = np.zeros((len(section_ys) + 1, 10))
      y_pairs[1:, 0::2] = y_pairs[:-1, 1::2] = section_ys

      # Slope for interpolating left and right sides
      slope = (section_ys[-1] - section_ys[0]).astype(np.double) \
            / (section_xs[-1] - section_xs[0])
      y_pairs[0, 0::2] = np.rint(section_ys[0] - slope*section_xs[0])
      y_pairs[-1, 1::2] = np.rint(section_ys[-1] \
                        + slope*(self.im.shape[1] - section_xs[-1]))

      # Expected y-intercepts for staff lines at each x value
      expected_y = np.zeros((5, self.im.shape[1]), dtype=int)
      for xs, ys in zip(x_pairs, y_pairs):
        y_mat = ys.reshape(5, 2)
        mat = np.array([np.linspace(1.0, 1.0/(xs[1] - xs[0]), xs[1]-xs[0]),
        np.linspace(0, 1 - 1.0/(xs[1]-xs[0]), xs[1]-xs[0])])
        self.line_positions[i, :, xs[0]:xs[1]] = y_mat.dot(mat).astype(int)
    self.page.staff_line_positions = self.line_positions

  def mask_staff_lines(self):
    # Expect area above and below staff line to be empty when masking
    xs = np.tile(np.arange(self.im.shape[1]), len(self.staves)*5) \
           .reshape(self.line_positions.shape)
    im_above = self.im[self.line_positions - self.staff_thick, xs]
    im_below = self.im[self.line_positions + self.staff_thick, xs]
    mask_staff, mask_line, line_xs = np.where((im_below == 0) & (im_above == 0))
    line_ys = self.line_positions[mask_staff, mask_line, line_xs]
    mask_ys = line_ys.ravel()[:, np.newaxis].repeat(2*self.staff_thick+1, axis=1)
    mask_ys[:] += np.linspace(-self.staff_thick, self.staff_thick,
                              2*self.staff_thick + 1)
    mask_xs = line_xs.ravel().repeat(2*self.staff_thick + 1)
    self.page.im[mask_ys.ravel(), mask_xs] = 0
    self.page.staff_mask = np.vstack((mask_ys.ravel(), mask_xs))

  def color_image(self):
    # Gray out masked staff lines
    colored_array = np.array(self.colored)
    colored_array[self.page.staff_mask[0], self.page.staff_mask[1]] |= 0x80
    self.page.colored = self.colored = Image.fromarray(colored_array)
    for staff in self.staves:
      staff.draw(self.colored)
    # Draw staff cross-sections
    d = ImageDraw.Draw(self.colored)
    for x, y1, y2, y3, y4, y5 in self.sections:
      for y in (y1, y2, y3, y4, y5):
        d.ellipse((x-3, y-3, x+3, y+3), outline=0, fill=255)

  def process(self):
    self.get_spacing()
    self.sections = np.zeros((0, 6), dtype=np.double) # x, y1, ...
    self.search_staff_intervals()
    self.build_staves()
    self.interpolate_staff_line_positions()
    self.mask_staff_lines()
    print self.staves
