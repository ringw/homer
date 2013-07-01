import numpy as np
import scipy.cluster.hierarchy
import ImageDraw

class Staff:
  def __init__(self):
    self.lines = tuple([] for i in range(5))
  def add_point(self, x, ys):
    # Insert points in ys into each line at position x
    for i in range(5):
      y = ys[i]
      line = self.lines[i]
      line_ind = 0
      line_len = len(line)
      while line_ind < line_len:
        if line[line_ind] >= x:
          break
        line_ind += 1
      line.insert(line_ind, (x, y))
  def draw(self, im):
    d = ImageDraw.Draw(im)
    for line in self.lines:
      d.line(line, fill=(255, 0, 0))
class StaffTask:
  def __init__(self, page):
    self.im = page.im
    self.colored = page.colored

  STAFF_SPACE_DY = 10
  STAFF_THICK_DY = 5

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

  def get_spacing(self):
    dark_cols = np.array(self.col_runs[:,3], dtype=bool)

    # Histogram of light lengths (space between staff lines)
    dists = np.bincount(self.col_runs[~dark_cols, 2])
    # Histogram of dark lengths (thickness of staff lines)
    thicks = np.bincount(self.col_runs[dark_cols, 2])

    self.staff_space = np.argmax(dists)
    self.staff_thick = np.argmax(thicks)
    return (self.staff_space, self.staff_thick)

  # 1D array -> 5-tuples of coordinates of possible staff cross-sections
  def cross_sections(self, col_num):
    # Extract runs from this column
    runs = self.col_runs[self.col_runs[:, 0] == col_num]
    if runs.shape[0] < 9: return []
    # Ensure first run is dark
    if not runs[0,3]:
      runs = runs[1:]
    
    # Runs alternate color, so there are (len(runs) + 1)/2 - 4 candidates
    num_candidates = (runs.shape[0] + 1)/2 - 4
    candidate_starts = np.arange(0, 2*num_candidates, 2)
    # Consecutive 9-tuples of candidate staff cross-sections with run lengths
    # (dark, light, dark, ..., dark)
    sections = np.zeros((num_candidates, 9), dtype=int)
    for i in xrange(9):
      sections[:,i] = runs[candidate_starts + i, 2]
    thicks = sections[:,[0,2,4,6,8]]
    dists = sections[:,[1,3,5,7]]

    # Find indices of staff cross-sections
    candidates = (np.abs(np.mean(thicks, axis=1) - self.staff_thick) \
                    < self.STAFF_THICK_DY) \
                 & (np.abs(np.mean(dists, axis=1) - self.staff_space) \
                       < self.STAFF_SPACE_DY) \
                 & (np.std(thicks, axis=1) < 0.5) & (np.std(dists, axis=1) < 1.0)
    candidate_ind = np.arange(0, runs.shape[0])[candidates]
    # Find staff line center y from runs position and length
    centers = np.zeros((np.count_nonzero(candidates), 5))
    for i in xrange(5):
      centers[:,i] = runs[(candidate_ind + i)*2, 1] + runs[(candidate_ind + i)*2, 2]/2

    return map(tuple, centers)

  def get_staff_clusters(self):
    sections = []
    xs = []
    for x in self.sections:
      for section in self.sections[x]:
        sections.append(section)
        xs.append(x)
    cnums = scipy.cluster.hierarchy.fclusterdata(np.array(sections),
                    self.staff_space/2, criterion='distance', method='single')
    clusters = []
    for i in xrange(1, len(cnums)):
      cnum = cnums[i]
      while len(clusters) < cnum: clusters.append([])
      clusters[cnum - 1].append([xs[i]] + list(sections[i]))
    self.clusters = clusters

  CLUSTER_MIN_SIZE = 10
  def create_staves(self):
    # Clusters above minimum size become staves
    staves = []
    for cluster in self.clusters:
      if len(cluster) < self.CLUSTER_MIN_SIZE: continue
      staff = Staff()
      for points in cluster:
        staff.add_point(points[0], points[1:])
      staves.append(staff)
    self.staves = staves

  def color_image(self):
    for staff in self.staves:
      staff.draw(self.colored)

  def process(self):
    self.get_runlength_encoding()
    self.get_spacing()
    self.sections = dict()
    # Sum columns
    proj = self.im.sum(axis=0)
    hist = [[] for i in xrange(self.im.shape[1])]
    i = 0
    for col in proj:
      if i > 0 and proj[i-1] < col:
        i += 1
        continue
      if i + 1 < len(proj) and proj[i+1] < col:
        i += 1 
        continue
      hist[col].append(i)
      i += 1
    # Go through x-coordinates by amount of colored pixels
    # until we have enough detail to define staves
    i = 0
    for xs in hist:
      if i < 10:
        i += 1
        continue
      if i == 300: break # XXX
      for x in xs:
        self.sections[x] = self.cross_sections(x)
      i += 1
    #i = 0
    #for h in hist:
    #  print (i, h)
    #  i += 1
    print self.sections
    self.get_staff_clusters()
    self.create_staves()
    print self.staves
