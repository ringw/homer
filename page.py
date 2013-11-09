import numpy as np
import image
import rotate
import layout

class Page:
  def __init__(self, image_data):
    self.image_data = image_data
    self._im = None
    self._colored = None # RGB copy of image for coloring
    self.staves = []
    self.rotate = rotate.RotateTask(self)
    self.layout = layout.Layout(self)

  def load_image(self):
    self._im, self._colored = image.image_array(self.image_data)

  @property
  def im(self):
    if self._im is None:
      self.load_image()
    return self._im

  @property
  def colored(self):
    if self._colored is None:
      self.load_image()
    return self._colored

  def destroy_image(self):
    self._im = None
    self._colored = None

  # Store column and row runlength encoding
  def get_runlength_encoding(self):
    # rows of (col, start_ind, end_ind, length, is_dark)
    all_col_runs = [] 
    for col_num, col in enumerate(self.im.T):
      # Position of every last pixel in its run
      # Intentionally do not include first or last run
      pos, = np.where(col[1:] != col[:-1])
      lengths = np.diff(pos)
      col_runs = np.zeros((lengths.shape[0], 5), dtype=int)
      col_runs[:,0] = col_num
      col_runs[:,1] = pos[:-1] + 1
      col_runs[:,2] = pos[1:]
      col_runs[:,3] = lengths
      col_runs[:,4] = (col[pos[:-1] + 1] == 1)
      all_col_runs.append(col_runs)
    self.col_runs = np.concatenate(all_col_runs)
    all_row_runs = []
    for row_num, row in enumerate(self.im):
      pos, = np.where(row[1:] != row[:-1])
      lengths = np.diff(pos)
      row_runs = np.zeros((lengths.shape[0], 5), dtype=int)
      row_runs[:,0] = row_num
      row_runs[:,1] = pos[:-1] + 1
      row_runs[:,2] = pos[1:]
      row_runs[:,3] = lengths
      row_runs[:,4] = (row[pos[:-1] + 1] == 1)
      all_row_runs.append(row_runs)
    self.row_runs = np.concatenate(all_row_runs)

  def get_spacing(self):
    dark_cols = np.array(self.col_runs[:,4], dtype=bool)

    # Histogram of light lengths (space between staff lines)
    dists = np.bincount(self.col_runs[~dark_cols, 3])
    # Histogram of dark lengths (thickness of staff lines)
    thicks = np.bincount(self.col_runs[dark_cols, 3])

    if not any(dists) or not any(thicks):
      self.blank = True
      return None
    self.blank = False

    self.staff_space = np.argmax(dists)
    self.staff_thick = np.argmax(thicks)
    self.staff_dist  = self.staff_space + self.staff_thick
    return (self.staff_space, self.staff_thick)

  def process(self):
    self.rotate.process()
    self.layout.process()
  def color(self):
    pass

  def get_glyph(self, g):
    glyph_box = self.glyph_boxes[g]
    return (self.labels[glyph_box] == g+1).astype(int)
