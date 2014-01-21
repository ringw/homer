from .pagetree import PageTree
from ..util import label_1d, center_of_mass_1d
import logging
from numpy import *
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.ndimage import distance_transform_edt
from scipy.sparse.csgraph import dijkstra

class StaffBuilder:
  def __init__(self, page):
    self.page = page
    self.tree = page.tree

  def split_tree(self):
    self.tree.recursive_split(PageTree.can_split)

  def build_staves_from_clusters(self):
    tree_leaves = array(filter(lambda n: n.leaf, self.tree.traverse()))
    staff_leaves = array([bool(n.staff) for n in tree_leaves])
    staff_sections = [n.staff for n in tree_leaves[staff_leaves]]
    if not any(staff_sections):
      return None
    staff_sections = asarray(filter(None, staff_sections))
    dists = distance.pdist(staff_sections)
    Z = hierarchy.linkage(dists, method='complete')
    fc = hierarchy.fcluster(Z, sqrt(5) * self.page.staff_dist / 2,
                            criterion='distance')
    cluster_nums = argsort(bincount(fc))[::-1]
    self.staves = []
    seen_staff_ys = zeros(self.page.im.shape[0], dtype=bool)
    for c in cluster_nums:
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
    self.page.staves = self.staves = self.staves[argsort( self.staves[:,0] )]
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

  def assign_staves(self):
    # Assign staves to neighbors which don't have a staff
    unassigned = set(n for n in self.tree.traverse() if n.leaf and not n.staff)
    while len(unassigned) > 0:
      node = iter(unassigned).next()
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
        for n in stretch:
          if n.bounds[0] <= staff[0] and n.bounds[0]+n.bounds[2] >= staff[0]:
            n.staff = staff
            n.staff_num = staff_num
      unassigned.difference_update(stretch)
    # Update with previous and next staff
    # prev_staff: first staff number above this node
    # next_staff: first staff number in this node or any nodes beneath it
    for leaf in self.tree.leaves():
      for n in leaf.neighbors('n'):
        if n.staff:
          leaf.prev_staff = n.staff_num
          break
      else:
        leaf.prev_staff = -1
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

        # leaf's staff may hang down below leaf; update south neighbors
        if leaf.bounds[0] + leaf.bounds[2] <= leaf.staff[-1]:
          for s in leaf.nonleaf_neighbors('s'):
            if leaf.staff[-1] < s.bounds[0]:
              break
            for leaf2 in s.leaves():
              if leaf2.bounds[0] <= leaf.staff[-1]:
                leaf2.staff = leaf.staff
                leaf2.staff_num = leaf.staff_num
                leaf2.prev_staff = leaf.staff_num
                leaf2.post_staff = int(rint(leaf.staff[-1]))
      else:
        leaf.pre_staff = None
        leaf.post_staff = leaf.bounds[0]
        leaf.next_staff = self.staves.shape[0] # no staves after
        for s in leaf.neighbors('s'):
          if s.staff:
            leaf.next_staff = s.staff_num
            break
    # Now assign prev_staff and next_staff based on global staff positions
    for leaf in self.tree.leaves():
      if leaf.staff is None:
        for staff_num, staff in enumerate(self.page.staves):
          if leaf.bounds[0] <= staff[0]:
            leaf.prev_staff = staff_num - 1
            leaf.next_staff = staff_num
            break

  def mask_staff_ys(self):
    # Set staff pixels to INT_MAX
    INT_MAX = iinfo(self.page.im.dtype).max
    for l in self.tree.leaves():
      if l.staff:
        staff = rint(array(l.staff)).astype(int)
        for y in staff:
          xs = l.slice[1]
          mask = (self.page.im[y - self.page.staff_thick, xs] == 0) \
               & (self.page.im[y + self.page.staff_thick, xs] == 0)
          self.page.im[y - self.page.staff_thick:y + self.page.staff_thick,
                       where(mask)[0] + xs.start] *= INT_MAX

  def process(self):
    self.split_tree()
    self.build_staves_from_clusters()
    self.assign_staves()
    self.mask_staff_ys()

  def show(self):
    import pylab
    leaves = self.page.tree.leaves()
    for leaf in leaves:
      pylab.plot([leaf.bounds[1], leaf.bounds[1],
                  leaf.bounds[1]+leaf.bounds[3],
                  leaf.bounds[1]+leaf.bounds[3],
                  leaf.bounds[1]],
                 [leaf.bounds[0], leaf.bounds[0]+leaf.bounds[2],
                  leaf.bounds[0]+leaf.bounds[2],
                  leaf.bounds[0],
                  leaf.bounds[0]],
                 'g', alpha=0.1)
      if leaf.staff:
        midx = leaf.bounds[1] + leaf.bounds[3]/2
        pylab.plot(repeat(midx, 5), leaf.staff, 'g.', alpha=0.5)

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

class StaffSegmenter:
  def __init__(self, page):
    self.page = page
    self.tree = page.tree

  def leaf_boundary_points(self, ys, xs):
    image = self.page.im[ys, xs]
    if image.size == 0:
      return array([])
    proj = (image != 0).sum(1)
    # Detect runs of horizontal projection with various threshold
    max_pixels = 0
    # Find vertical runs where every horizontal slice is background
    points = []
    covered = zeros_like(proj, bool)
    while max_pixels < (xs.stop - xs.start)/2 and not covered.all():
      is_bg = proj <= max_pixels
      # Break up runs every staff_dist so that we have several possible points
      is_bg[arange(len(is_bg)) % self.page.staff_dist == 0] = 0
      if is_bg.any():
        if is_bg.all():
          break
        background_ys, num_ys = label_1d(is_bg)
        background_centers = center_of_mass_1d(background_ys)
        if len(background_centers):
          dy = self.page.staff_thick
          for center in background_centers:
            if not covered[rint(center).astype(int)]:
              points.append(center + ys.start)
            covered[max(center-dy, 0):min(center+dy, xs.stop-xs.start)] = True
      max_pixels += self.page.staff_thick
    if len(points) == 0:
      return array([(ys.start + ys.stop)/2])
    else:
      return unique(points)

  # Multiply each edge distance by an arbitrary cost function,
  # favoring larger distance transform along boundary
  def edge_cost(self, y0, x0, y1, x1):
    # Allow vector inputs which are cast to 1D array
    y0, x0, y1, x1 = map(atleast_1d, [y0, x0, y1, x1])
    # It should be that x1 > x0
    assert((x1 > x0).all())
    dx = amax(x1 - x0)
    # Create arrays long enough for all arguments
    ts = arange(dx)[None, :].repeat(y0.size, 0) # parametrize x and y
    ys = rint(ts * (y1 - y0)[:, None].astype(double)
                 / (x1 - x0)[:, None] + y0[:, None]).astype(int)
    xs = ts + x0[:, None]
    # Mask out bad ys and xs with 0
    in_range = ((ts < (x1 - x0)[:, None])
                & (0 <= ys) & (ys < self.page.im.shape[0])
                & (0 <= xs) & (xs < self.page.im.shape[1]))
    ys *= in_range
    xs *= in_range
    if not hasattr(self, 'distance_transform'):
      self.distance_transform = distance_transform_edt(self.page.im == 0)
    dists = self.distance_transform[ys, xs]
    count = (dists < self.page.staff_thick).astype(int)
    count *= in_range
    if y0.size == 1:
      return count.sum() + 1
    else:
      return count.sum(1) + 1

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
      return int(this_sum / this_count - self.page.staff_dist)
    elif staff == len(self.page.staves):
      return int(prev_sum / prev_count + self.page.staff_dist)
    else:
      return int((this_sum / this_count + prev_sum / prev_count) / 2)

  def build_boundary(self, staff):
    inter_y = self.inter_staff_y(staff)

    leaf_slices = []
    leaf_xs = []
    for leaf in self.tree.leaves():
      ys, xs = leaf.slice
      if leaf.prev_staff == staff - 1 and leaf.next_staff == staff:
        if leaf.staff:
          ys = slice(ys.start, leaf.staff[0] - self.page.staff_dist)
      elif staff > 0 and leaf.next_staff == staff - 1 and leaf.staff:
        # a node may be assigned to the previous staff, but contain some
        # area of the boundary
        ys = slice(leaf.staff[-1] + self.page.staff_dist, ys.stop)
      else:
        continue
      leaf_slices.append((ys.start, ys.stop, xs.start, xs.stop))
      leaf_xs.append((xs.start + xs.stop)/2)
    # sort staff_leaves and leaf_slices by center x coordinate
    inds = argsort(leaf_xs)
    leaf_slices = array(leaf_slices)
    leaf_slices = leaf_slices[inds]
    leaf_xs = take(leaf_xs, inds)
    leaf_ys = [self.leaf_boundary_points(slice(y0, y1), slice(x0, x1))
               for (y0, y1, x0, x1) in leaf_slices]
    num_per_leaf = [len(a) for a in leaf_ys]
    point_xs = leaf_xs.repeat(num_per_leaf)
    point_ys = concatenate(leaf_ys)
    point_leaf = arange(len(leaf_ys)).repeat(num_per_leaf)
    point_leaf_bounds = leaf_slices[point_leaf]
    # Calculate distance from start to each point
    # (0/no connection if point does not belong to a leftmost leaf)
    from_start = repeat(0, len(point_ys))
    is_leftmost = point_leaf_bounds[:, 2] == amin(point_leaf_bounds[:, 2])
    num_start = count_nonzero(is_leftmost)
    leftmost_dists = sqrt((point_ys[is_leftmost] - inter_y)**2
                          + (point_xs[is_leftmost] - 0)**2)
    leftmost_y0 = repeat(inter_y, num_start)
    leftmost_x0 = repeat(0, num_start)
    leftmost_cost = self.edge_cost(leftmost_y0, leftmost_x0,
                                   point_ys[is_leftmost], point_xs[is_leftmost])
    leftmost_dists *= leftmost_cost
    from_start[is_leftmost] = leftmost_dists
    # Build condensed distance matrix.
    # Additional "start" node with index 0, and "end" node with last index.
    # Both have y-index inter_y and x-indices the edge of the image.

    # Create all pairs of points
    # Each point, after appearing as the first point in the pair, is followed
    # by its distance from the end node
    # This method of generating pairs of indices is fast but not at all
    # elegant. Is there a better way to do it?
    pnums = arange(len(point_ys))
    numinpair = len(point_ys) - pnums
    p0 = pnums.repeat(numinpair)
    p1 = arange(1, len(p0) + 1)
    # Need to adjust elements of p1 downwards
    adj = zeros_like(p1)
    adj[cumsum(numinpair[:-1])] = -numinpair[:-1] + 1
    p1 += cumsum(adj)
    # Indicate distance to end with -1
    # also prevents complaining when we use p1 to index into something
    p1[p1 == len(point_ys)] = -1
    bounds0 = point_leaf_bounds[p0]
    bounds1 = point_leaf_bounds[p1]
    good_pairs = bounds0[:, 3] == bounds1[:, 2]
    # Must replace pairs where p1 == -1 with whether p1's leaf is rightmost
    good_pairs[p1 == -1] = bounds0[p1 == -1, 3] == amax(bounds0[p1 == -1, 3])
    x0, y0 = point_xs[p0], point_ys[p0]
    x1, y1 = point_xs[p1], point_ys[p1]
    x1[p1 == -1] = self.page.im.shape[1]
    y1[p1 == -1] = inter_y

    x0 = x0[good_pairs]
    y0 = y0[good_pairs]
    x1 = x1[good_pairs]
    y1 = y1[good_pairs]
    pdists = zeros(len(p0), double)
    pdists[good_pairs] = (sqrt((y1 - y0)**2 + (x1 - x0)**2)
                          * self.edge_cost(y0, x0, y1, x1))

    distlist = concatenate([from_start, [0], pdists])
    # We traverse our path in reverse order, so we want the lower triangular.
    distmat = tril(distance.squareform(distlist), 1)

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
      path.append((int(rint(point_ys[next_vert-1])),
                   int(rint(point_xs[next_vert-1]))))
      last_vert = next_vert
    path.append((inter_y, self.page.im.shape[1]))
    return path

  # Build region_labels array
  def build_region_labels(self):
    self.region_labels = region_labels = ones_like(self.page.im, dtype=uint8)
    region_labels *= 255 # -1 in uint8
    y_index = arange(self.page.im.shape[0])[:, None]
    # Increment by 1 for each boundary any pixel lies beneath
    for boundary in self.boundaries:
      # y-index of boundary for each x-index
      boundary_y = zeros(self.page.im.shape[1], int)
      for (y0, x0), (y1, x1) in zip(boundary[:-1], boundary[1:]):
        dx = arange(x1 - x0)
        y = dx * float(y1 - y0) / (x1 - x0) + y0
        boundary_y[x0:x1] = rint(y)
      mask = y_index >= boundary_y[None, :]
      region_labels[mask] += 1
    region_labels[region_labels == len(self.page.staves) + 1] = 255

  def process(self):
    self.boundaries = []
    for i in xrange(0, len(self.page.staves) + 1):
      try:
        self.boundaries.append(self.build_boundary(i))
      except:
        inter_y = self.inter_staff_y(i)
        self.boundaries.append([(inter_y, 0), (inter_y, self.page.im.shape[1])])
        logging.warn('Boundary building failed')
    self.page.boundaries = self.boundaries
    self.build_region_labels()

  def show(self):
    import pylab
    for b in self.boundaries:
      for p0, p1 in zip(b[:-1], b[1:]):
        pylab.plot([p0[1], p1[1]], [p0[0], p1[0]], 'y')
