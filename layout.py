from quadtree import Quadtree
from numpy import *
from scipy.cluster import hierarchy
from scipy.ndimage import label, center_of_mass, distance_transform_edt
from scipy.spatial import Voronoi, distance
from scipy.sparse.csgraph import dijkstra

class PageTree(Quadtree):
  def __init__(self, page, bounds=None, parent=None):
    if bounds is None:
      bounds = (0, 0) + page.im.shape
    Quadtree.__init__(self, bounds, parent=parent)
    self.page = page
    # False for 0 staves, 5-tuple for single staff
    self.staff = None

  def create_child(self, bounds):
    return PageTree(self.page, bounds=bounds, parent=self)

  def analyze(self):
    pass

  def can_split(self):
    if self.bounds[2] <= self.page.staff_dist*4:
      return False
    if self.leaf and self.staff is None:
      ys, xs = self.slice
      # Extend slice to include possible staff starting at bottom
      staff_dist = self.page.staff_thick + self.page.staff_space
      ys = slice(ys.start, min(ys.stop + staff_dist*4, self.page.im.shape[0]))
      staff_sections = self.page.im[ys, xs].sum(1)
      if staff_sections.sum() < self.bounds[2] * self.bounds[3] / 100.0:
        return False # section is empty
      expected = amax(staff_sections) * 0.75
      if expected < (xs.stop - xs.start)/2:
        return True
      staff_ys = staff_sections >= expected
      y_ind, = where(staff_ys)
      if len(y_ind) == 0:
        return True
      # Combine consecutive y_ind
      new_run = concatenate([[0], diff(y_ind) > 1]).astype(int)
      run_ind = cumsum(new_run) # Index starting from 0 for each positive run
      num_in_run = bincount(run_ind)
      if any(num_in_run) >= self.page.staff_space*1.5:
        return True
      run_centers = bincount(run_ind, weights=y_ind).astype(double) / num_in_run
      # Remove anything past the actual rect except a staff starting here
      if len(run_centers) >= 5 \
         and run_centers[0] < self.bounds[2] \
         and any(run_centers[1:5] >= self.bounds[2]):
        run_centers = run_centers[:5]
      else:
        run_centers = run_centers[run_centers < self.bounds[2]]
        if len(run_centers) != 5:
          return True
      # Look for actual staff
      staff_dists = diff(run_centers)
      if (abs(mean(staff_dists) - staff_dist) < self.page.staff_thick
          and std(staff_dists) <= self.page.staff_thick):
        self.staff = tuple(run_centers + self.bounds[0])
        return False
      else:
        return True
    return False

  def recursive_split(self):
    if self.try_split():
      self.nw.recursive_split()
      self.sw.recursive_split()
      self.ne.recursive_split()
      self.se.recursive_split()

class Layout:
  def __init__(self, page):
    self.page = page

  def build_tree(self):
    self.tree = PageTree(self.page)
    self.tree.recursive_split()

  def build_staves(self):
    # Split horizontal stretches of quadtree until we have only leaves
    horizontals = [[self.tree]]
    while True:
      new_horizontals = []
      did_split = False
      for run in horizontals:
        run_top = []
        run_bottom = None
        for n in run:
          if n.leaf:
            run_top.append(n)
            if run_bottom is not None:
              run_bottom.append(n)
          else:
            if run_bottom is None:
              run_bottom = list(run_top)
            run_top.append(n.nw)
            run_top.append(n.ne)
            run_bottom.append(n.sw)
            run_bottom.append(n.se)
            did_split = True
        if run_bottom is not None:
          new_horizontals.append(run_top)
          new_horizontals.append(run_bottom)
        else:
          new_horizontals.append(run)
      if not did_split:
        break
      horizontals = new_horizontals
    self.horizontals = filter(lambda r: any([n.staff for n in r]), horizontals)
    self.build_staves_from_clusters()

  def build_staves_from_clusters(self):
    tree_leaves = array(filter(lambda n: n.leaf, self.tree.traverse()))
    staff_leaves = array([bool(n.staff) for n in tree_leaves])
    staff_sections = [n.staff for n in tree_leaves[staff_leaves]]
    if not any(staff_sections):
      return None
    staff_sections = asarray(filter(None, staff_sections))
    dists = distance.pdist(staff_sections)
    Z = hierarchy.linkage(dists, method='complete')
    fc = hierarchy.fcluster(Z, sqrt(5) * (self.page.staff_space + self.page.staff_thick) / 2, criterion='distance')
    cluster_nums = argsort(bincount(fc))[::-1]
    self.staves = []
    seen_staff_ys = zeros(self.page.im.shape[0], dtype=bool)
    for c in cluster_nums:
      staff_num = len(self.staves)
      which_nodes = fc == c
      if not which_nodes.any(): break
      sections = staff_sections[which_nodes]
      staff_ys = rint(sections.mean(axis=0)).astype(int)
      if any(seen_staff_ys[staff_ys[0]:staff_ys[-1]]):
        # This cluster overlaps with a staff that's already been seen
        # Likely off by 1 staff line or other issue
        continue
      staff_ymin = max(staff_ys[0] - self.page.staff_dist*3, 0)
      staff_ymax = min(staff_ys[-1] + self.page.staff_dist*3,
                       self.page.im.shape[0])
      seen_staff_ys[staff_ymin:staff_ymax] = True
      self.staves.append(staff_ys)
    self.staves = asarray(self.staves, dtype=double)
    self.staves = self.staves[argsort( self.staves[:,0] )]
    # Assign tree leaves to staves which are close enough, or remove
    # staff assignment if it is too far from any detected staves
    for leaf in tree_leaves[staff_leaves]:
      FOUND_STAFF = False
      for i, staff in enumerate(self.staves):
        if amax(abs(leaf.staff - staff)) < self.page.staff_dist / 2.0:
          leaf.staff_num = i
          FOUND_STAFF = True
          break
      if not FOUND_STAFF:
        leaf.staff = None

  def update_staves(self):
    # Assign staves to neighbors which don't have as staff
    unassigned = filter(lambda n: n.leaf and not n.staff, self.tree.traverse())
    while len(unassigned) > 0:
      node = unassigned[0]
      stretch = []
      left_staff = None
      for left_node in node.neighbors('w'):
        if left_node.staff:
          left_staff = left_node
          break
        else:
          stretch.insert(0, left_node)
      stretch.append(node)
      right_staff = None
      for right_node in node.neighbors('e'):
        if right_node.staff:
          right_staff = right_node
          break
        else:
          stretch.append(right_node)
      if left_staff and right_staff \
          and left_staff.staff_num == right_staff.staff_num:
        # XXX: weighted average?
        staff = tuple((array(left_staff.staff) + array(right_staff.staff)) / 2)
        staff_num = left_staff.staff_num
      elif left_staff:
        staff = left_staff.staff
        staff_num = left_staff.staff_num
      elif right_staff:
        staff = right_staff.staff
        staff_num = right_staff.staff_num
      else:
        staff = None
        staff_num = None
      for n in stretch:
        if staff and (n.bounds[0] <= staff[-1] and n.bounds[0]+n.bounds[2] >= staff[0]):
          n.staff = staff
          n.staff_num = staff_num
        if n in unassigned:
          unassigned.remove(n)
    # Update with previous and next staff
    for leaf in self.tree.leaves():
      if leaf.staff:
        start = int(rint(leaf.staff[0]))
        stop = int(rint(leaf.staff[-1]))
        leaf.pre_staff = (start
                          if start >= leaf.bounds[0]
                          else None)
        leaf.post_staff = (stop
                           if stop < leaf.bounds[0]+leaf.bounds[2]
                           else None)
        # next_staff: next staff after leaf.bounds[0]
        leaf.next_staff = leaf.staff_num
      else:
        leaf.pre_staff = None
        leaf.post_staff = leaf.bounds[0]
        # Invariant: if we go down far enough, we will intersect with the
        # next staff
        # XXX: we may want to unset staff on margins so this may not hold
        leaf.next_staff = self.staves.shape[0] # no staves after
        for s in leaf.neighbors('s'):
          if s.staff:
            leaf.next_staff = s.staff_num
            break

  def mask_staff_ys(self):
    for l in self.tree.leaves():
      if l.staff:
        staff = rint(array(l.staff)).astype(int)
        for y in staff:
          xs = l.slice[1]
          mask = (self.page.im[y - self.page.staff_thick, xs] == 0) \
               & (self.page.im[y + self.page.staff_thick, xs] == 0)
          self.page.im[y - self.page.staff_thick:y + self.page.staff_thick,
                       where(mask)[0] + xs.start] *= -1

  def leaf_boundary_points(self, ys, xs):
    #ys, xs = leaf.slice
    #if leaf.staff:
      #ys = slice(ys.start, leaf.staff[0] - self.page.staff_dist)
    image = self.page.im[ys, xs]
    # Find vertical runs where every horizontal slice is background
    is_bg = (image != 0).sum(1) < self.page.staff_thick
    if not is_bg.any():
      return zeros(0) # empty array
    background_ys, num_ys = label(is_bg)
    background_centers = array(center_of_mass(is_bg, background_ys, arange(1, num_ys + 1)))
    return (background_centers.ravel() + ys.start)

  # Multiply each edge distance by an arbitrary cost function
  # (in this case, sum of distance transform of image along line)
  def edge_cost(self, y0, x0, y1, x1):
    # It should be that x1 > x0
    assert(x1 > x0)
    dx = x1 - x0
    ys = rint(arange(dx) * (y1 - y0) / float(x1 - x0) + y0).astype(int)
    xs = arange(x0, x1, 1 if x0 < x1 else -1)
    if not hasattr(self, 'distance_transform'):
      self.distance_transform = distance_transform_edt(self.page.im == 0)
    #  self.distance_transform[self.distance_transform < self.page.staff_thick] = -self.page.staff_dist/4.0
    #return exp(-self.distance_transform[ys, xs]/self.page.staff_dist*4).sum()
    return count_nonzero(self.distance_transform[ys, xs] < self.page.staff_thick) + 1

  # Given staff number, return a small distance above staff (for first staff), 
  # a small distance below (for last staff), or midway between staff
  # and previous staff
  def inter_staff_y(self, staff):
    prev_sum = 0
    prev_count = 0
    this_sum = 0
    this_count = 0
    for leaf in self.tree.leaves():
      if leaf.staff:
        if leaf.next_staff == staff - 1:
          prev_sum += leaf.staff[-1]
          prev_count += 1
        elif leaf.next_staff == staff:
          this_sum += leaf.staff[0]
          this_count += 1
    if staff == 0:
      return this_sum / this_count - self.page.staff_dist
    elif staff == len(self.staves):
      return prev_sum / prev_count + self.page.staff_dist
    else:
      return (this_sum / this_count + prev_sum / prev_count) / 2

  def build_boundary(self, staff):
    leaf_slices = []
    leaf_xs = []
    for leaf in self.tree.leaves():
      ys, xs = leaf.slice
      if leaf.next_staff == staff:
        if leaf.staff:
          ys = slice(ys.start, leaf.staff[0] - self.page.staff_dist)
      elif staff > 0 and leaf.next_staff == staff - 1 and leaf.staff:
        # a node may be assigned to the previous staff, but contain some
        # area of the boundary
        ys = slice(leaf.staff[-1] + self.page.staff_dist, ys.stop)
      else:
        continue
      leaf_slices.append((ys, xs))
      leaf_xs.append((xs.start + xs.stop)/2)
    # sort staff_leaves and leaf_slices by center x coordinate
    inds = argsort(leaf_xs)
    leaf_slices = array(leaf_slices)
    leaf_slices = leaf_slices[inds]
    leaf_xs = take(leaf_xs, inds)
    leaf_ys = [self.leaf_boundary_points(ys, xs) for (ys, xs) in leaf_slices]
    num_per_leaf = [len(a) for a in leaf_ys]
    point_xs = leaf_xs.repeat(num_per_leaf)
    point_ys = concatenate(leaf_ys)
    point_leaf = arange(len(leaf_ys)).repeat(num_per_leaf)
    # Build condensed distance matrix, storing distance from start
    # separately and then concatenating.
    # Distance to end is stored after all entries for the point.
    from_start = []
    distlist = []
    # Additional "start" node with index 0, and "end" node with last index
    # Both have y-index inter_y and x-indices the edge of the image.
    inter_y = self.inter_staff_y(staff)
    for p0 in xrange(len(point_ys)):
      leaf0num = point_leaf[p0]
      x0, y0 = point_xs[p0], point_ys[p0]
      x0min = leaf_slices[leaf0num, 1].start
      x0max = leaf_slices[leaf0num, 1].stop

      # Add distance from start
      from_start.append(sqrt((y0 - inter_y)**2 + (x0 - 0)**2)
                        * self.edge_cost(inter_y, 0, y0, x0)
                        if x0min == 0 else inf)

      for p1 in xrange(p0+1, len(point_ys)):
        leaf1num = point_leaf[p1]
        x1min = leaf_slices[leaf1num, 1].start
        x1max = leaf_slices[leaf1num, 1].stop
        if (x0max == x1min) or (x1max == x0min):
          x1, y1 = point_xs[p1], point_ys[p1]
          distlist.append(sqrt((y0 - y1)**2 + (x0 - x1)**2)
                          * self.edge_cost(y0, x0, y1, x1))
        else:
          distlist.append(inf)

      # Add distance to end
      xmax = self.page.im.shape[1]
      distlist.append(sqrt((y0 - inter_y)**2 + (x0 - xmax)**2)
                      * self.edge_cost(y0, x0, inter_y, xmax)
                      if x0max == xmax else inf)

    # Note: start is not connected to end.
    distlist = concatenate([from_start, [inf], distlist])
    # We traverse our path in reverse order, so we want the lower triangular.
    distmat = tril(distance.squareform(distlist), 1)
    distmat[isnan(distmat)] = 0

    dists, paths = dijkstra(distmat, return_predecessors=True)
    self.distmat, self.dists, self.paths = distmat, dists, paths

    # Build up path from paths
    path = [(inter_y, 0)]
    last_vert = start = 0
    end = distmat.shape[0] - 1
    while True:
      next_vert = paths[end, last_vert]
      if next_vert == end:
        break
      path.append((point_ys[next_vert-1], point_xs[next_vert-1]))
      last_vert = next_vert
    path.append((inter_y, self.page.im.shape[1]))
    return path

  def build_boundaries(self):
    self.boundaries = []
    # XXX: this only makes sense for piano solo music and we should detect
    # which staves are actually joined
    for i in xrange(0, self.staves.shape[0]+1, 2):
      self.boundaries.append(self.build_boundary(i))

  def process(self):
    self.build_tree()
    self.build_staves_from_clusters()
    self.update_staves()
    self.mask_staff_ys()
    self.build_boundaries()

  # Simple check to ensure we didn't miss any staves
  def check_missing_staves(self):
    col_runs = self.page.col_runs.copy()
    # Remove runs overlapping with any detected staff
    to_remove = zeros(col_runs.shape[0], bool)
    for staff in self.staves:
      to_remove |= ((col_runs[:,1] >= staff[0] - self.page.staff_dist)
                    & (col_runs[:,2] <= staff[-1] + self.page.staff_dist))
    col_runs = col_runs[~ to_remove]
    hist = bincount(col_runs[:, 3])
    # If all staves are removed, we expect no staves will remain
    # A staff that goes across the whole page should have ~ page.shape[1]*5 runs
    return not (hist[10:] >= self.page.im.shape[1]*5.0/2).any()

  def show(self, show_grid=True, show_boundaries=True):
    leaves = filter(lambda n: n.leaf, self.tree.traverse())
    import pylab
    if show_grid:
      for leaf in leaves:
        pylab.plot([leaf.bounds[1], leaf.bounds[1],
                    leaf.bounds[1]+leaf.bounds[3],
                    leaf.bounds[1]+leaf.bounds[3],
                    leaf.bounds[1]],
                   [leaf.bounds[0], leaf.bounds[0]+leaf.bounds[2],
                    leaf.bounds[0]+leaf.bounds[2],
                    leaf.bounds[0],
                    leaf.bounds[0]],
                   'g', alpha=0.25)

    if show_boundaries:
      for b in self.boundaries:
        for p0, p1 in zip(b[:-1], b[1:]):
          pylab.plot([p0[1], p1[1]], [p0[0], p1[0]], 'y')
