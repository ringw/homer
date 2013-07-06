import numpy as np

class Page:
  def __init__(self, im, colored=None):
    self.im = im
    self.colored = colored # RGB copy of image for coloring
    self.staves = []
    self.get_runlength_encoding()

  # Store column runlength encoding
  def get_runlength_encoding(self):
    # Lists of per-column runs
    runs = [] # column -> ndarray of rows of (col, start_ind, length, is_dark)
    for col_num, col in enumerate(self.im.T):
      # Position of every last pixel in its run
      # Intentionally do not include first or last run
      pos, = np.diff(col).nonzero()
      lengths = np.diff(pos)
      col_runs = np.zeros((lengths.shape[0], 4), dtype=int)
      col_runs[:,0] = col_num
      col_runs[:,1] = pos[:-1] + 1
      col_runs[:,2] = lengths
      col_runs[:,3] = (col[pos[:-1] + 1] == 1)
      runs.append(col_runs)
    self.col_runs = np.concatenate(runs)
