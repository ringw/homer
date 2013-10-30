from quadtree import Quadtree
from numpy import *
from scipy.ndimage import convolve1d

class PageTree(Quadtree):
  def __init__(self, page, bounds=None, parent=None):
    if bounds is None:
      bounds = (0, 0) + page.im.shape
    Quadtree.__init__(self, bounds, parent=parent)
    self.page = page
    # False for 0 staves, 5-tuple for single staff
    self.staff = None

  def create_child(self, bounds):
    return PageTree(self.page, bounds=bounds, parent=self)

  def analyze(self):
    pass

  def can_split(self):
    if self.bounds[2] <= self.page.staff_space*2:
      return False
    if self.leaf and self.staff is None:
      ys, xs = self.slice
      # Extend slice to include possible staff starting at bottom
      staff_dist = self.page.staff_thick + self.page.staff_space
      ys = slice(ys.start, min(ys.stop + staff_dist*6, self.page.im.shape[0]))
      staff_sections = self.page.im[ys, xs].sum(1)
      expected = (xs.stop - xs.start) / 2
      staff_ys = staff_sections >= expected
      y_ind, = where(staff_ys)
      if len(y_ind) == 0:
        return True
      # Combine consecutive y_ind
      new_run = concatenate([[0], diff(y_ind) > 1]).astype(int)
      run_ind = cumsum(new_run) # Index starting from 0 for each positive run
      num_in_run = bincount(run_ind)
      if any(num_in_run) >= self.page.staff_space*1.5:
        return True
      run_centers = bincount(run_ind, weights=y_ind).astype(double) / num_in_run
      # Remove anything past the actual rect except a staff starting here
      if len(run_centers) > 5 \
         and run_centers[0] < self.bounds[2] \
         and any(run_centers[1:5] >= self.bounds[2]):
        run_centers = run_centers[:5]
      else:
        run_centers = run_centers[run_centers < self.bounds[2]]
        if len(run_centers) != 5:
          return True
      # Look for actual staff
      staff_dists = diff(run_centers)
      if (abs(mean(staff_dists) - staff_dist) < self.page.staff_thick
          and std(staff_dists) <= self.page.staff_thick):
        self.staff = tuple(run_centers + self.bounds[0])
        return False
      else:
        return True
    return False

  def recursive_split(self):
    if self.try_split():
      self.nw.recursive_split()
      self.sw.recursive_split()
      self.ne.recursive_split()
      self.se.recursive_split()

class Layout:
  def __init__(self, page):
    self.page = page

  def build(self):
    self.tree = PageTree(self.page)
    self.tree.recursive_split()

def searchstaves(q):
  if q.leaf:
    if type(q.staff) is tuple:
      return [q.staff]
    else:
      return []
  else:
    A = []
    for subq in [q.nw, q.sw, q.ne, q.se]:
      A.extend(searchstaves(subq))
    return A
