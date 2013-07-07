import numpy as np

class Page:
  def __init__(self, im, colored=None):
    self.im = im
    self.colored = colored # RGB copy of image for coloring
    self.staves = []
    self.get_runlength_encoding()

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
