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

  # 2-tuple (distance, thickness)
  def staff_spacing(self, arr):
    # Histograms of white band size (distance) and black band size (thickness)
    dists = np.zeros(arr.shape[0] + 1, dtype=int)
    thicks = np.zeros(arr.shape[0] + 1, dtype=int)
    for c in arr.transpose():
      isBlack = False
      run = 0
      for pixel in c:
        if bool(pixel) != isBlack:
          if isBlack == True:
            thicks[run] += 1
          else:
            dists[run] += 1
          run = 0
          isBlack = bool(pixel)
        else:
          run += 1
    return (np.argmax(dists), np.argmax(thicks))

  # 1D array -> 5-tuples of coordinates of possible staff cross-sections
  def get_cross_sections(self, arr):
    staves = []
    # Track previous acceptable staff members, thickness, and average distance
    # Throw away previous coordinates until standard deviation is small enough
    coords = []
    thick = []
    dists = []
    startOn = None # None if last pixel was off, or position of last start
    i = 0
    for pixel in arr:
      if startOn is None:
        if pixel > 0:
          startOn = i
      elif pixel == 0:
        point = (i - 1 + startOn) / 2
        if len(coords) == 5:
          dists = [dists[1], dists[2], dists[3], point - coords[4]]
          coords = [coords[1], coords[2], coords[3], coords[4], point]
          thick = [thick[1], thick[2], thick[3], thick[4], i - startOn]
        else:
          if len(coords) > 0:
            dists.append(point - coords[-1])
          coords.append(point)
          thick.append(i - startOn)

        if len(coords) == 5 \
          and abs(np.mean(dists) - self.staff_space) < self.STAFF_SPACE_DY \
          and abs(np.mean(thick) - self.staff_thick) < self.STAFF_THICK_DY \
          and np.std(dists) < 2.0 and np.std(thick) < 1.0:
          staves.append(tuple(coords))
        startOn = None
      i += 1
    return staves

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
    self.staff_space, self.staff_thick = self.staff_spacing(self.im)
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
      if i == 300: break
      for x in xs:
        self.sections[x] = self.get_cross_sections(self.im[:,x])
      i += 1
    #i = 0
    #for h in hist:
    #  print (i, h)
    #  i += 1
    self.get_staff_clusters()
    #print self.clusters
    #print len(self.clusters)
    #print map(len, self.clusters)
    #for c in self.clusters:
    #  print np.array(c).T
    self.create_staves()
    print self.staves
