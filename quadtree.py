directions = dict(n=0, s=0, w=1, e=1)
opposites = dict(n='s', s='n', w='e', e='w')

class Quadtree:
  # bounds: (y, x, h, w)
  def __init__(self, bounds, parent=None):
    self.bounds = bounds
    self.parent = parent
    self.nw = self.ne = self.sw = self.se = None

  # Look up path to root
  def path(self):
    if self.parent is None:
      return []
    else:
      return self.parent.path() + [self.direction]

  @property
  def slice(self):
    return (slice(self.bounds[0], self.bounds[0] + self.bounds[2]),
            slice(self.bounds[1], self.bounds[1] + self.bounds[3]))

  @property
  def direction(self):
    # Assign nw/ne/sw/se to this node based on parent
    return (     None if self.parent is None
            else 'nw' if self.parent.nw == self
            else 'ne' if self.parent.ne == self
            else 'sw' if self.parent.sw == self
            else 'se')

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
    return (self.nw == None) # if nw is None, there are no children

  def can_split(self):
    return self.leaf
  def try_split(self):
    if self.can_split():
      return self.split()
    return False

  # Return child with path (e.g. ["ne", "sw", "se"])
  # If path asks for a child of a leaf node, the leaf is returned
  # If path does not end at a leaf, all leaf children are returned in a list
  def child(self, path):
    if self.leaf:
      return self
    elif len(path) == 0:
      return filter(lambda n: n.leaf, self.traverse())
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

  # Return lists (n, e, s, w) with all neighbors
  def neighbors(self, direction=None):
    if direction is None:
      return tuple(self.neighbors(d) for d in ['n', 'e', 's', 'w'])

    neighs = []
    n = self.neighbor(direction)
    path = self.path()
    ind = directions[direction]
    while n is not None:
      # Find subchildren of n along our path
      children = [n]
      while True:
        nonleaf = False
        new_children = []
        for child in children:
          if child.leaf:
            new_children.append(child)
          else:
            nonleaf = True
            depth = len(child.path())
            if depth >= len(path):
              new_children.extend(filter(lambda node: node.leaf, child.traverse()))
            else:
              next_subpath = path[len(child.path())]
              if ind == 0:
                new_children.append(getattr(child, opposites[direction] + next_subpath[1]))
                new_children.append(getattr(child, direction + next_subpath[1]))
              else:
                new_children.append(getattr(child, next_subpath[0] + opposites[direction]))
                new_children.append(getattr(child, next_subpath[0] + direction))
            
        if nonleaf:
          children = new_children
        else:
          break
      neighs.extend(children)
      n = n.neighbor(direction)
    return neighs


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
  def leaves(self):
    return filter(lambda n: n.leaf, self.traverse())

  # Return starting x-coordinates of all subtrees
  def x_coords(self):
    import numpy
    xs = numpy.zeros(self.bounds[3], bool)
    for leaf in self.leaves():
      xs[leaf.bounds[1]] = True
    return xs
