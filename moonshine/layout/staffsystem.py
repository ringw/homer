from .. import hough
import numpy as np

class SystemBuilder:
  def __init__(self, page):
    self.page = page

  def process(self):
    # Incremental search of staves
    self.next_staff = 0
    self.num_staves = 2
    self.staff_systems = []
    while self.build_next_system():
      if self.next_staff >= len(self.page.staves):
        break
    else: # while terminated due to build_next_system failing
      raise ValueError("Failed to find a staff system")

  def system_barlines(self, start, count):
    start_y = self.page.staves[start, 0]
    end_y   = self.page.staves[start + count-1, -1]
    inner_space = self.page.im[start_y:end_y].T
    if not hasattr(self, 'thetas'):
      self.thetas = np.linspace(-0.02*np.pi, 0.02*np.pi, 25)
    hough_transform = hough.hough_line(inner_space, self.thetas)
    hough_lines = hough.hough_peaks(hough_transform,
                     size=(2*self.page.staff_dist, len(self.thetas)),
                     min_val=inner_space.shape[1]*0.75)
    if hough_lines.size == 0:
      return None
    else:
      return sorted(list(hough_lines[:, 0]))

  def build_next_system(self):
    num_staves = self.num_staves
    barlines = self.system_barlines(self.next_staff, num_staves)
    if barlines is None:
      # No barlines; try removing staves from the bottom until we find barlines
      # which go across all staves
      while barlines is None and num_staves - 1 > 0:
        num_staves -= 1
        barlines = self.system_barlines(self.next_staff, num_staves)
    else:
      # Add additional staves until we don't have any barlines crossing them all
      num_staves += 1
      while self.next_staff + num_staves < len(self.page.staves):
        new_barlines = self.system_barlines(self.next_staff, num_staves)
        if new_barlines is None:
          break
        else:
          barlines = new_barlines
          num_staves += 1
      num_staves -= 1

    if barlines is not None:
      self.staff_systems.append((self.next_staff, self.next_staff + num_staves,
                                 barlines))
      self.next_staff += num_staves
      self.num_staves = num_staves
      return True

  def show(self):
    import pylab
    for start, stop, barlines in self.staff_systems:
      y0 = self.page.staves[start, 0]
      y1 = self.page.staves[stop-1, -1]
      for x in barlines:
        pylab.plot([x, x], [y0, y1], 'g')
