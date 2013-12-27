import numpy as np
from scipy.ndimage import label, center_of_mass

class MeasureBuilder:
  def __init__(self, page):
    self.page = page
    self.tree = page.tree

  def process(self):
    for staff in xrange(len(self.page.staves)):
      self.build_barlines(staff)

  def build_barlines(self, staff):
    staff_image = self.page.im.copy()
    staff_image[self.page.staff_segmenter.region_labels != staff] = 0
    non_staff = staff_image == 1
    barline = non_staff.sum(0) >= self.page.staff_dist * 4
    barline_labels, num_barlines = label(barline)
    bar_x = center_of_mass(barline, barline_labels,
                           arange(1, num_barlines+1))[:,0]
