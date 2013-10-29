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

  def can_split(self):
    print 'hi'
    if self.leaf and self.staff is None:
      staff_sections = self.page.staff_filter[self.slice].sum(1)
      maximum = self.page.filter_max * self.bounds[3]
      #indices, = where(staff_sections > maximum * 1.0/4)
      indices, = where(staff_sections > maximum * 1.0/8)
      print indices
      return (diff(indices) > 1).any()
    return False

  def recursive_split(self):
    if self.try_split():
      self.nw.try_split()
      self.sw.try_split()
      self.ne.try_split()
      self.se.try_split()

class Layout:
  def __init__(self, page):
    self.page = page

  @property
  def weights(self):
    space = self.page.staff_space
    thick = self.page.staff_thick
    weights = concatenate((-2*space*ones(thick),
                           -thick*ones(space),
                           2*space*ones(thick),
                           -thick*ones(space),
                           2*space*ones(thick),
                           -thick*ones(space),
                           2*space*ones(thick),
                           -thick*ones(space),
                           2*space*ones(thick),
                           -thick*ones(space),
                           2*space*ones(thick),
                           -thick*ones(space),
                           -2*space*ones(thick)))
    return (weights, abs(weights).sum())

  def analyze(self):
    weights, maximum = self.weights
    signed_im = self.page.im.astype(float32) * 2 - 1
    self.page.filter_max = maximum
    staff_filter = convolve1d(signed_im, weights, axis=0)
    #orig = staff_filter.copy()
    #space = self.page.staff_space
    #thick = self.page.staff_thick
    #staff_filter[:-(space+thick)] -= orig[:-(space+thick)]/2
    #staff_filter[space+thick:] -= orig[:-(space+thick)]/2
    self.page.staff_filter = staff_filter

  def build(self):
    self.tree = PageTree(self.page)
    self.tree.recursive_split()
