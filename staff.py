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
    self.im = page.im
    self.colored = page.colored

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

  def find_center_ys(self):
    mask, offset = self.convolution_filter()
    cv = convolve(self.im.sum(1), mask, mode='same')
    cv[cv < 0] = 0
    ys = []
    STAFF_DIST = 5*(self.page.staff_space + self.page.staff_thick)
    while count_nonzero(cv):
      peak = argmax(cv)
      ys.append(peak)
      cv[max(0, peak-STAFF_DIST):min(len(cv)-1, peak+STAFF_DIST)] = 0
    return asarray(ys)

  def color_image(self):
    # Gray out center ys
    colored_array = array(self.colored)
    center_ys = self.find_center_ys()
    to_gray = zeros(self.page.im.shape[0], dtype=bool)
    to_gray[(arange(-5, 5)[:, None] + center_ys[None, :]).ravel()] = True
    colored_array[to_gray] |= 0x80
    self.page.colored = self.colored = Image.fromarray(colored_array)

  def process(self):
    print self.find_center_ys()
