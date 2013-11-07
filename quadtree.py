class Quadtree:
  # bounds: (y, x, h, w)
  def __init__(self, bounds, parent=None):
    self.bounds = bounds
    self.parent = parent
    self.nw = self.ne = self.sw = self.se = None

  @property
  def slice(self):
    return (slice(self.bounds[0], self.bounds[0] + self.bounds[2]),
            slice(self.bounds[1], self.bounds[1] + self.bounds[3]))

  def create_child(self, bounds):
    return Quadtree(bounds, parent=self)

  def split(self):
    if not self.leaf:
      raise ValueError("Attempt to split non-leaf")
    elif self.bounds[2] > 1 and self.bounds[3] > 1:
      self.nw = self.create_child((self.bounds[0], self.bounds[1],
                                   self.bounds[2]/2, self.bounds[3]/2))
      south_y = self.bounds[0] + self.nw.bounds[2]
      south_h = self.bounds[0] + self.bounds[2] - south_y
      east_x = self.bounds[1] + self.nw.bounds[3]
      east_w = self.bounds[1] + self.bounds[3] - east_x
      self.sw = self.create_child((south_y, self.bounds[1],
                                   south_h, self.bounds[3]/2))
      self.ne = self.create_child((self.bounds[0], east_x,
                                   self.bounds[2]/2, east_w))
      self.se = self.create_child((south_y, east_x,
                                   south_h, east_w))
      return True
    else:
      return False

  @property
  def leaf(self):
    return (self.nw == None)

  def can_split(self):
    return self.leaf
  def try_split(self):
    if self.can_split():
      return self.split()
    return False

  def __repr__(self):
    return "<%s%s%s at %s>" % (self.__class__.__name__, self.bounds, " (leaf)" if self.leaf else "", hex(id(self)))

  def traverse(self):
    if self.leaf:
      return [self]
    A = []
    A.extend(self.nw.traverse())
    A.extend(self.ne.traverse())
    A.extend(self.sw.traverse())
    A.extend(self.se.traverse())
    return A
