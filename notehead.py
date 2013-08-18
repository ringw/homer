import numpy as np
from scipy.signal import convolve2d


def ellipse_candidate_centers(im, gradient, radius_max=40):
  edge_points = np.column_stack(np.where(im))
  ts = gradient[tuple(edge_points.T)]
  dists = np.arange(0, radius_max + 1)
  points = edge_points[..., np.newaxis].repeat(len(dists), axis=2)
  for p in xrange(len(points)):
    points[p, 0] += dists * np.sin(ts[p])
    points[p, 1] += dists * np.cos(ts[p])
  good = np.where((points[:, 0] >= 0) & (points[:, 0] < im.shape[0])
                  & (points[:, 1] >= 0) & (points[:, 1] < im.shape[1]))
  points[:, 0] *= im.shape[1]
  bins = np.sum(points, axis=1)[good]
  counts = np.bincount(bins)
  counts.resize(np.prod(im.shape))
  counts = counts.reshape(im.shape)
  import matplotlib.pyplot as plt
  colored = np.zeros(im.shape + (3,))
  colored[..., 0] = counts
  colored[..., 2] = im
  colored[..., 0] /= np.amax(colored[..., 0])
  colored[..., 2] /= np.amax(colored[..., 2])
  #plt.clf()
  #plt.imshow(colored)
  #plt.show()
  y0,x0 = (None,None)
  while np.count_nonzero(counts):
    y,x = np.unravel_index(np.argmax(counts), im.shape)
    if y0 is None:
      y0 = y
      x0 = x
    #print y,x
    # Invalidate nearby points
    counts[max(0, y-radius_max):y+radius_max, max(0, x-radius_max):x+radius_max] = 0
  return (y0,x0)

class NoteheadsTask:
  def __init__(self, page):
    self.page = page

  # Find single "model" glyph (e.g. isolated quarter note) to extract
  # parameters of notehead ellipse (which should be invariant under translation)
  def choose_model_glyphs(self):
    # Search for glyphs which are large enough to contain a notehead
    glyph_size = self.page.glyph_bounds[..., 1] - self.page.glyph_bounds[..., 0]
    glyphs = np.all(glyph_size > self.page.staff_space, axis=1)
    glyphs &= glyph_size[:, 1] < self.page.staff_space*3
    glyphs &= glyph_size[:, 0] > self.page.staff_space*4
    glyphs &= glyph_size[:, 0] < self.page.staff_space*6
    glyph_nums, = np.where(glyphs)
    glyph_bounds = self.page.glyph_bounds[glyphs]

    individual_glyphs = [
        (self.page.labels[glyph_bounds[i,0,0]:glyph_bounds[i,0,1],
                          glyph_bounds[i,1,0]:glyph_bounds[i,1,1]] == (g+1))
        .astype(int) for i,g in enumerate(glyph_nums)
    ]
    # The vertical projection of an ellipse is modeled by a circle
    # whose projection is sqrt(1 - x^2). We concatenate the square of
    # the vertical projection for each glyph and take the finite difference,
    # and look for short lines with a small negative slope.
    # (d/dx (1 - x^2) = -2x)
    projections = [np.sum(glyph, axis=0) for glyph in individual_glyphs]
    glyph_proj_num = np.repeat(glyph_nums, map(len, projections))
    glyph_proj = np.concatenate(projections)
    glyph_proj **= 2
    self.model_glyph_proj = glyph_proj
    self.model_glyph_proj_num = glyph_proj_num
    self.model_glyph_mask = glyphs
    self.model_glyphs = individual_glyphs

  def search_model_glyph_vertical_projections(self):
    dy = np.empty_like(self.model_glyph_proj)
    dy[[0, -1]] = 0
    dy[1:-1] = self.model_glyph_proj[2:] - self.model_glyph_proj[:-2]
    self.glyph_proj_deriv = dy
    xs = np.arange(self.page.staff_space)
    ys = xs[:, None] + np.arange(len(dy) - len(xs))[None, :]
    fit, res, rank, sing, rcond = np.polyfit(xs, dy[ys], 1, full=True)
    # Candidates must have small negative slope and intersect near center
    # of interval
    #print '\n'.join(map(str, fit.T))
    candidates = (fit[0] < -self.page.staff_space/2) & (fit[0] > -2*self.page.staff_space)
    denom = (-fit[0]).astype(np.double)
    denom[denom == 0] = 1e-10
    local_maximum = fit[1]/denom
    candidates &= (np.abs(local_maximum - len(xs)/2) < len(xs)/4)
    # Local maximum should actually have a tangent near 0
    maximum_point = np.rint(local_maximum).astype(int) + np.arange(ys.shape[1])
    maximum_point[~candidates] = 0
    candidates &= np.abs(dy[maximum_point]) < self.page.staff_thick
    self.proj_fit = fit
    self.proj_residual = res
    self.proj_candidates = candidates
    proj_x = np.argmin(res[candidates])
    self.candidate_glyph = self.model_glyph_proj_num[candidates][proj_x]
    glyph_proj_start = np.where(self.model_glyph_proj_num[candidates] == self.candidate_glyph)[0][0]
    self.candidate_x = local_maximum[candidates][proj_x] + proj_x - glyph_proj_start
#    import pylab as P
#    glyph = (self.page.labels[self.page.glyph_boxes[self.candidate_glyph]]
#             == (self.candidate_glyph + 1)).astype(int)
#    im = np.zeros(glyph.shape + (3,), dtype=np.uint8)
#    im[..., 2] = glyph * 255
#    im[:, int(np.rint(self.candidate_x)), 0] = 255
#    P.imshow(im)
#    P.show()
    return (self.candidate_glyph, self.candidate_x)

  def model_ellipse_glyph(self):
    self.choose_model_glyphs()
    self.search_model_glyph_vertical_projections()

  def process(self):
    self.model_ellipse_glyph()

  def color_image(self):
    pass
