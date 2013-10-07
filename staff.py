from numpy import *
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

class StavesTask:
  def __init__(self, page):
    self.page = page

  # Filter to be convolved with horizontal projection of image,
  # along with offset for actual staff
  def convolution_filter(self):
    space = -self.page.staff_thick * ones(self.page.staff_space)
    staff =  self.page.staff_space * ones(self.page.staff_thick)
    mask = concatenate((space,   staff, space,
                               2*staff, space,
                               5*staff, space,
                               2*staff, space,
                                 staff, space))
    return mask, 3*len(space) + len(staff)*5/2

  def find_candidate_center_ys(self):
    mask, offset = self.convolution_filter()
    cv = convolve(self.page.im.sum(1), mask, mode='same')
    cv[cv < 0] = 0
    ys = []
    STAFF_DIST = 5*(self.page.staff_space + self.page.staff_thick)
    while count_nonzero(cv):
      peak = argmax(cv)
      ys.append(peak)
      cv[max(0, peak-STAFF_DIST):min(len(cv)-1, peak+STAFF_DIST)] = 0
    return asarray(ys)

  def find_staff_ys(self):
    candidates = self.find_candidate_center_ys()
    SEARCH_SIZE = 3*(self.page.staff_space + self.page.staff_thick)
    hsum = self.page.im.sum(1)
    near_candidate_indices = (arange(-SEARCH_SIZE, SEARCH_SIZE+1)[None]
                              + candidates[:, None])
    near_candidate_indices[near_candidate_indices < 0] = 0
    near_candidate_indices[near_candidate_indices >= self.page.im.shape[0]] \
        = self.page.im.shape[0] - 1
    NUM_NEAR = near_candidate_indices.shape[1]
    near_candidates = hsum[near_candidate_indices]
    INVALIDATE_SIZE = self.page.staff_space / 2
    near_candidates[:, SEARCH_SIZE+1 - INVALIDATE_SIZE
                      :SEARCH_SIZE+1 + INVALIDATE_SIZE] = 0
    lines = zeros((len(candidates), 5), dtype=int)
    lines[:, 0] = candidates
    for i in range(1, 5):
      next_line = argmax(near_candidates, axis=1)
      lines[:, i] = next_line + candidates - SEARCH_SIZE - 1
      near_candidates[  (near_candidate_indices
                         >= (lines[:, i] - INVALIDATE_SIZE)[:, None])
                      & (near_candidate_indices
                         <= (lines[:, i] + INVALIDATE_SIZE)[:, None])] = 0
    # Filter out unlikely staves
    lines.sort(axis=1)
    line_dist = diff(lines, axis=1)
    staves = ((line_dist.std(axis=1) < self.page.staff_thick)
              & (abs(line_dist.mean(axis=1)
                     - (self.page.staff_space + self.page.staff_thick))
                 < self.page.staff_thick))
    lines = lines[staves]
    self.staff_ys = lines
    return lines[argsort(lines[:, 0])]

  def mask_staff_ys(self):
    self.page.staves = []
    # Index into image around each staff line
    STAFF_MASK_SIZE = (self.page.staff_space + self.page.staff_thick) / 2
    for staff in self.staff_ys:
      staffObject = Staff()
      staffObject.line_masks = []
      for line in staff:
        y_slice = slice(line - STAFF_MASK_SIZE,
                        line + STAFF_MASK_SIZE + 1)
        section = self.page.im[y_slice]
        to_mask = sum(section, 0) < (self.page.staff_thick * 2)
        self.page.im[y_slice, to_mask] = 0
        line_mask = zeros_like(self.page.im, dtype=bool)
        line_mask[y_slice, to_mask] = True
        staffObject.line_masks.append(where(line_mask & self.page.im))
      staffObject.add_point(0, staff)
      staffObject.add_point(self.page.im.shape[1]-1, staff)
      self.page.staves.append(staffObject)

  def color_image(self):
    # Gray out center ys
    colored_array = array(self.page.colored)
    for staff in self.page.staves:
      for line_mask in staff.line_masks:
        colored_array[line_mask[1], line_mask[0]] |= 0x80
    self.page.colored = self.page.colored = Image.fromarray(colored_array)
    for staff in self.page.staves:
      staff.draw(self.page.colored)

  def process(self):
    print self.find_staff_ys()
    self.mask_staff_ys()
