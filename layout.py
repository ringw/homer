from quadtree import Quadtree
from numpy import *
from scipy.cluster import hierarchy
from scipy.ndimage import label, center_of_mass
from scipy.spatial import Voronoi, distance
from scipy.sparse.csgraph import shortest_path

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
      #print horizontals
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

  def build_boundaries(self):
    labels, nlabels = label(self.page.im == 1)
    centers = center_of_mass(self.page.im == 1, labels, arange(1, nlabels + 1))
    staffpoints = set()
    for leaf in self.tree.leaves():
      if leaf.staff:
        y0 = leaf.staff[0]
        y1 = leaf.staff[1]
        x0 = leaf.bounds[1]
        x1 = leaf.bounds[1] + leaf.bounds[3]
        staffpoints.add((y0, x0))
        staffpoints.add((y0, x1))
        staffpoints.add((y1, x0))
        staffpoints.add((y1, x1))
    staffpoints = array(list(staffpoints), double)
    self.voronoi = Voronoi(vstack((centers, staffpoints)))

  def choose_boundaries(self):
    all_ridges = array(self.voronoi.ridge_vertices)
    i = 8
    #for i in xrange(self.staves.shape[0]+1):
    verts = ((self.voronoi.vertices[:,0] > self.staves[i-1, -1])
             & (self.voronoi.vertices[:,0] < self.staves[i, 0]))
    vertinds, = where(verts)
    vertconvert = cumsum(verts) - 1
    vertpoints = self.voronoi.vertices[verts]
    self.vertpoints = vertpoints
    ok_ridges = (in1d(all_ridges[:,0], vertinds)
                 & in1d(all_ridges[:,1], vertinds))
    ridges = vertconvert[all_ridges[ok_ridges]]
    ridge_mat = zeros((len(vertpoints), len(vertpoints)), bool)
    ridge_mat[ridges[:,0], ridges[:,1]] = True
    ridge_mat[ridges[:,1], ridges[:,0]] = True
    self.ridges, self.ridge_mat = ridges, ridge_mat
    # XXX: these are not very good predictions
    start = argmin(vertpoints[:, 1])
    end = argmax(vertpoints[:, 1])
    self.start, self.end = start,end
    distmat = distance.squareform(distance.pdist(vertpoints))
    distmat[~ridge_mat] = 1e10
    dists, paths = shortest_path(distmat, return_predecessors=True)
    self.paths = paths

    # Build up path from paths
    path = [vertpoints[start]]
    self.path = path
    last_vert = start
    while True:
      next_vert = paths[end, last_vert]
      path.append(vertpoints[next_vert])
      if next_vert == end:
        break
      last_vert = next_vert

    # Shortest path with Voronoi of object centers is not very good with long slurs
    # For each leaf, if we detect a single long, near-horizontal line
    # between these staves, move the boundary just above it
    for leaf in self.tree.leaves():
      if leaf.next_staff != i:
        continue
      runs = self.page.im[leaf.slice].sum(1) >= leaf.bounds[3]/2
      if leaf.pre_staff:
        runs[:leaf.pre_staff - leaf.bounds[0]] = 0
      if leaf.post_staff:
        runs[leaf.post_staff - leaf.bounds[0]:] = 0
      # Ensure there is only one run
      indices, = where(runs)
      if (len(indices) and (diff(indices) > 1).all()
          and len(indices) < self.page.staff_thick*2):
        new_y = leaf.bounds[0] + int(mean(indices)) - self.page.staff_thick*2
        ind = 0
        while ind < len(path) and path[ind][1] < leaf.bounds[1]:
          ind += 1
        while ind < len(path) and path[ind][1] < leaf.bounds[1]+leaf.bounds[3]:
          path.pop(ind)
        path.insert(ind, array([new_y, leaf.bounds[1]]))
        path.insert(ind+1, array([new_y, leaf.bounds[1]+leaf.bounds[3] - 1]))

  def process(self):
    self.build_tree()
    self.build_staves_from_clusters()
    self.update_staves()
    self.mask_staff_ys()
    self.build_boundaries()
    self.choose_boundaries()

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

  def show(self):
    leaves = filter(lambda n: n.leaf, self.tree.traverse())
    import pylab
    for leaf in leaves:
      pylab.plot([leaf.bounds[1], leaf.bounds[1], leaf.bounds[1]+leaf.bounds[3],
                  leaf.bounds[1]+leaf.bounds[3], leaf.bounds[1]],
                 [leaf.bounds[0], leaf.bounds[0]+leaf.bounds[2],
                  leaf.bounds[0]+leaf.bounds[2], leaf.bounds[0], leaf.bounds[0]]
                 , 'g', alpha=0.25)

    for p0, p1 in zip(self.path[:-1], self.path[1:]):
      pylab.plot([p0[1], p1[1]], [p0[0], p1[0]], 'y')
