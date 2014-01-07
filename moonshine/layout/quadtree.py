import numpy as np

directions = dict(n=0, s=0, w=1, e=1)
opposites = dict(n='s', s='n', w='e', e='w')

class QuadTree:
  # bounds: (y, x, h, w)
  def __init__(self, bounds, parent=None):
    self.bounds = bounds
    self.parent = parent
    self.nw = self.ne = self.sw = self.se = None
    self.leaf = True
    # Parent must assign this node's direction
    self.direction = None

  def root(self):
    if self.parent is None:
      return self
    else:
      return self.parent.root()

  def path(self):
    ''' Look up path to root '''
    if self.parent is None:
      return []
    else:
      return self.parent.path() + [self.direction]

  # Look up leaf containing point (y, x)
  def find(self, y, x):
    if ((y < self.bounds[0]) or (self.bounds[0] + self.bounds[2] <= y)
        or (x < self.bounds[1]) or (self.bounds[1] + self.bounds[3] <= x)):
      return None
    elif self.leaf:
      return self
    else:
      # Generator searching each child
      search = (child.find(y, x) for child in
                [self.nw, self.ne, self.sw, self.se])
      return next((s for s in search if s is not None), None)

  @property
  def slice(self):
    return (slice(self.bounds[0], self.bounds[0] + self.bounds[2]),
            slice(self.bounds[1], self.bounds[1] + self.bounds[3]))

  def create_child(self, bounds):
    return QuadTree(bounds, parent=self)

  def split(self):
    if not self.leaf:
      raise ValueError("Attempt to split non-leaf")
    elif self.bounds[2] > 1 and self.bounds[3] > 1:
      self.nw = self.create_child((self.bounds[0], self.bounds[1],
                                   self.bounds[2]/2, self.bounds[3]/2))
      self.nw.direction = 'nw'
      south_y = self.bounds[0] + self.nw.bounds[2]
      south_h = self.bounds[0] + self.bounds[2] - south_y
      east_x = self.bounds[1] + self.nw.bounds[3]
      east_w = self.bounds[1] + self.bounds[3] - east_x
      self.sw = self.create_child((south_y, self.bounds[1],
                                   south_h, self.bounds[3]/2))
      self.sw.direction = 'sw'
      self.ne = self.create_child((self.bounds[0], east_x,
                                   self.bounds[2]/2, east_w))
      self.ne.direction = 'ne'
      self.se = self.create_child((south_y, east_x,
                                   south_h, east_w))
      self.se.direction = 'se'
      self.leaf = False
      return True
    else:
      return False

  def can_split(self):
    return self.leaf

  def recursive_split(self, criterion):
    if criterion(self):
      self.split()
      self.nw.recursive_split(criterion)
      self.sw.recursive_split(criterion)
      self.ne.recursive_split(criterion)
      self.se.recursive_split(criterion)

  # Return child with path (e.g. ["ne", "sw", "se"])
  def child(self, path):
    if self.leaf or len(path) == 0:
      return self
    else:
      return getattr(self, path[0]).child(path[1:])

  def neighbor(self, direction):
    ind = directions[direction]
    if self.parent is None:
      return None
    elif self.direction[ind] == direction:
      # We need to look at parent
      pneigh = self.parent.neighbor(direction)
      if not pneigh or pneigh.leaf:
        return pneigh
      elif ind == 0:
        return getattr(pneigh, opposites[direction] + self.direction[1])
      else:
        return getattr(pneigh, self.direction[0] + opposites[direction])
    else:
      if ind == 0:
        return getattr(self.parent, direction + self.direction[1])
      else:
        return getattr(self.parent, self.direction[0] + direction)

  def nonleaf_neighbors(self, direction):
    # ind = 0 (n/s) or 1 (e/w)
    ind = directions[direction]
    if self.parent is None:
      return
    elif self.direction[ind] == direction:
      # Get all of parent's neighbors, then try to get children along our path
      pneigh = self.parent.nonleaf_neighbors(direction)
      for p in pneigh:
        if p.leaf:
          yield p
        else:
          # Return 2 children along our path in the correct order
          child1 = list(self.direction)
          child1[ind] = opposites[direction]
          child1 = ''.join(child1)
          child2 = self.direction
          yield getattr(p, child1)
          yield getattr(p, child2)
    else:
      # Get first neighbor, then yield from their neighbors
      child = list(self.direction)
      child[ind] = direction
      child = ''.join(child)
      neigh = getattr(self.parent, child)
      yield neigh
      for n in neigh.nonleaf_neighbors(direction):
        yield n

  def neighbors(self, direction):
    ''' All leaves in the region bounded by self extended along direction '''
    for nonleaf in self.nonleaf_neighbors(direction):
      for leaf in nonleaf.leaves(direction):
        yield leaf

  def __repr__(self):
    return "<%s%s%s at %s>" % (self.__class__.__name__, self.bounds, " (leaf)" if self.leaf else "", hex(id(self)))

  def traverse(self):
    if self.leaf:
      yield self
      return
    for child in [self.nw, self.ne, self.sw, self.se]:
      for node in child.traverse():
        yield node

  def leaves(self, direction='s'):
    ''' Iterate over tree, returning all leaves. If direction is specified,
        the returned leaves will be ordered in a particular direction. '''
    if self.leaf:
      yield self
      return
    ind = directions[direction]
    child_leaves = [n.leaves(direction)
                    for n in [self.nw, self.ne, self.sw, self.se]]
    next_leaves = [None for l in child_leaves]
    while True:
      for iterator in child_leaves:
        ind = child_leaves.index(iterator)
        if next_leaves[ind] is None:
          try:
            next_leaves[ind] = iterator.next()
          except StopIteration:
            del child_leaves[ind]
            del next_leaves[ind]
            continue
      if len(child_leaves) == 0:
        break
      if direction == 's':
        order = [n.bounds[0] for n in next_leaves]
      elif direction == 'n':
        order = [-n.bounds[0] for n in next_leaves]
      elif direction == 'e':
        order = [n.bounds[1] for n in next_leaves]
      elif direction == 'w':
        order = [-n.bounds[1] for n in next_leaves]
      ind = np.argmin(order)
      yield next_leaves[ind]
      next_leaves[ind] = None

  def draw_rect(self):
    import pylab
    pylab.plot([self.bounds[1], self.bounds[1],
                self.bounds[1]+self.bounds[3], self.bounds[1]+self.bounds[3]],
               [self.bounds[0], self.bounds[0]+self.bounds[2],
                self.bounds[0]+self.bounds[2], self.bounds[0]], 'g')

  def draw(self):
    for leaf in self.leaves():
      leaf.draw_rect()
