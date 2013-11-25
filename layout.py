from quadtree import Quadtree
from numpy import *
from scipy.cluster import hierarchy
from scipy.spatial import distance

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
    if self.bounds[2] <= self.page.staff_dist*2:
      return False
    if self.leaf and self.staff is None:
      ys, xs = self.slice
      # Extend slice to include possible staff starting at bottom
      staff_dist = self.page.staff_thick + self.page.staff_space
      ys = slice(ys.start, min(ys.stop + staff_dist*4, self.page.im.shape[0]))
      staff_sections = self.page.im[ys, xs].sum(1)
      expected = (xs.stop - xs.start) * 2.0 / 3
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
      staff_ys = sections.mean(axis=0)
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
    
  def process(self):
    self.build_tree()
    self.build_staves_from_clusters()

  # Simple check to ensure we didn't miss any staves
  def check_missing_staves(self):
    col_runs = self.page.col_runs.copy()
    # Remove runs overlapping with any detected staff
    to_remove = zeros(col_runs.shape[0], bool)
    for staff in self.staves:
      to_remove |= ((col_runs[:,1] >= staff[0] - self.page.staff_thick)
                    & (col_runs[:,2] <= staff[-1] + self.page.staff_thick))
    col_runs = col_runs[~ to_remove]
    hist = bincount(col_runs[:, 3])
    # If all staves are removed, we expect no staves will remain
    # A staff that goes across the whole page should have ~ page.shape[1]*5 runs
    return not (hist[10:] >= self.page.im.shape[1]*5.0/2).any()
