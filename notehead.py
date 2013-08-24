import numpy as np
from debug import DEBUG
from scipy import ndimage

def ellipse_general_to_standard(A, B, C, D, E, F):
  """
  Convert an ellipse in general form:
    Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
  To standard form:
    ((x - x0) cos t - (y - y0) sin t)^2/a^2
    + ((x - x0) sin t + (y - y0) cos t)^2/b^2 = 1
  The ellipse has center (x0, y0), major and minor semi-axes a and b,
  and the angle to the semi-major axis is t.

  Parameters: A, B, C, D, E, F
  Returns: x0, y0, a, b, t
  """
  A, B, C, D, E, F = map(float, [A, B, C, D, E, F])
  # Matrix representation of conic section
  AQ = np.array([[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, F]])
  A33 = AQ[0:2, 0:2]
  # Formula for center
  x0, y0 = np.linalg.inv(A33).dot([-D/2, -E/2])
  # Each eigenvector of A33 lies along one of the axes
  evals, evecs = np.linalg.eigh(A33)
  # Semi-axes from reduced canonical equation
  a, b = np.sqrt(-np.linalg.det(AQ)/(np.linalg.det(A33)*evals))
  # Return major axis as "a" and angle to major axis between -pi/2 and pi/2
  t = np.arctan2(evecs[1,0], evecs[0,0]) # angle of axis "a"
  if b > a:
    a, b = b, a
    t += np.pi/2
  if   t < -np.pi/2: t += np.pi
  elif t >  np.pi/2: t -= np.pi
  return (x0, y0, a, b, t)

class NoteheadsTask:
  def __init__(self, page):
    self.page = page

  # Find single "model" glyph (e.g. isolated quarter note) to extract
  # parameters of notehead ellipse (which should be invariant under translation)
  def choose_model_glyph_candidates(self):
    # Search for glyphs which are large enough to contain a notehead
    glyph_size = self.page.glyph_bounds[..., 1] - self.page.glyph_bounds[..., 0]
    glyphs = np.all(glyph_size > self.page.staff_space, axis=1)
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
    self.candidate_glyph_proj = glyph_proj
    self.candidate_glyph_proj_num = glyph_proj_num
    self.candidate_glyph_mask = glyphs
    self.candidate_glyphs = individual_glyphs

  def choose_model_glyphs(self):
    dy = np.empty_like(self.candidate_glyph_proj)
    dy[[0, -1]] = 0
    dy[1:-1] = self.candidate_glyph_proj[2:] - self.candidate_glyph_proj[:-2]
    xs = np.arange(self.page.staff_space)
    ys = xs[:, None] + np.arange(len(dy) - len(xs))[None, :]
    fit, res, rank, sing, rcond = np.polyfit(xs, dy[ys], 1, full=True)
    # Candidates must have small negative slope and intersect near center
    # of interval
    candidates = (fit[0] < -self.page.staff_space/2) & (fit[0] > -2*self.page.staff_space)
    # Find intersection of fitted line at local maximum of projection
    # (predicted x-center of notehead)
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
    # Choose candidate xs which must have a residual below RES_MAX
    # (may include multiple x values for the same glyph)
    RES_MAX = float(len(xs)) * 1000 # XXX
    res[~candidates] = RES_MAX + 1
    candidate_xs = np.argsort(res)
    bad_xs, = np.where(res[candidate_xs] > RES_MAX)
    if len(bad_xs):
      candidate_xs = candidate_xs[:bad_xs[0]]
    # Choose model xs from candidates which are the best candidate
    # for each unique glyph
    #model_glyphs = self.candidate_glyph_proj_num[candidate_xs]
    model_glyphs, model_proj_xs = np.unique(self.candidate_glyph_proj_num[candidate_xs], return_index=True)
    # Index model_proj_xs back into candidate_glyph_proj_num
    model_proj_xs = candidate_xs[model_proj_xs]
    # Find actual x for each glyph from model_proj_xs
    #model_glyphs = self.candidate_glyph_proj_num[model_proj_xs]
    model_xs = model_proj_xs.copy()
    candidate_glyphs, glyph_proj_starts = np.unique(self.candidate_glyph_proj_num, return_index=True)
    glyph_num_start = np.zeros(self.page.num_glyphs, dtype=int)
    glyph_num_start[candidate_glyphs] = glyph_proj_starts
    # Index of model_proj_xs into candidate_glyph_proj_num
    candidate_nums = np.where(candidate_xs)
    model_xs -= glyph_num_start[model_glyphs]
    model_xs += local_maximum[model_xs]
    # Keep only sane values of model_xs
    xs_sane = ((0 <= model_xs)
               & (model_xs < np.diff(self.page.glyph_bounds[model_glyphs, 1])[0]))
    self.model_glyphs = model_glyphs[xs_sane]
    self.model_xs = model_xs[xs_sane]
    return (self.model_glyphs, self.model_xs)

  def fit_model_ellipses(self):
    self.model_ellipses = []
    for candidate_glyph, candidate_x in zip(self.model_glyphs, self.model_xs):
      # Extract border of model notehead
      glyph_box = self.page.glyph_boxes[candidate_glyph]
      glyph_section = self.page.labels[glyph_box]
      gnum = candidate_glyph + 1
      glyph = (glyph_section == gnum).astype(int)
      proj = np.sum(glyph, axis=0) ** 2
      glyph_border = glyph & (ndimage.convolve(glyph,
                                               [[1,1,1], [1,0,1], [1,1,1]],
                                               mode='constant') < 8)

      # Choose smooth sections of vertical projection for least squares fit,
      # based on second derivative of the Gaussian
      deriv = ndimage.gaussian_filter(proj, 1, order=2)
      xs = np.zeros_like(proj, dtype=bool)
      X_DIST = self.page.staff_space/2
      X0 = int(np.rint(candidate_x))
      xs[max(X0 - X_DIST, 0):X0 + X_DIST] = True
      xs &= np.abs(deriv) < self.page.staff_space

      if DEBUG():
        import pylab as P
        im = np.zeros(glyph.shape + (3,), dtype=bool)
        im[..., 2] = glyph
        im[:, xs, 0] = True
        im[:, X0, 1] = True
        P.imshow(im)
        P.show()

      # Get all border x and y coordinates for least squares fit
      y_coords, x_coords = np.where(glyph_border & xs[None, :])
      # Algorithm adapted from:
      # http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.7559
      D1 = np.column_stack((x_coords**2, x_coords*y_coords, y_coords**2))
      D2 = np.column_stack((x_coords, y_coords, np.ones(len(x_coords))))
      S1 = D1.astype(np.double).T.dot(D1)
      S2 = D1.astype(np.double).T.dot(D2)
      S3 = D2.astype(np.double).T.dot(D2)
      T = -np.linalg.inv(S3).dot(S2.T)
      M = S1 + S2.dot(T) # reduced scatter matrix
      M = np.vstack([M[2] / 2.0, -M[1], M[0] / 2.0]) # premultiply by inv(C1)
      evals, evecs = np.linalg.eig(M)
      cond = 4 * (evecs[0] * evecs[2]) - evecs[1]**2 # a^T . C . a
      a1 = evecs[:, np.where(cond > 0)[0][0]]
      params = np.concatenate((a1, T.dot(a1)))
      x0, y0, a, b, t = ellipse_general_to_standard(*params)
      self.model_ellipses.append((x0, y0, a, b, t))
      if DEBUG():
        import pylab as P
        im = np.zeros(glyph.shape + (3,), dtype=bool)
        im[..., 2] = glyph
        # Draw ellipse
        draw_ts = np.linspace(0, 2*np.pi, 100)
        draw_xs = x0 + a*np.cos(draw_ts)*np.cos(t) - b*np.sin(draw_ts)*np.sin(t)
        draw_ys = y0 + a*np.cos(draw_ts)*np.sin(t) + b*np.sin(draw_ts)*np.cos(t)
        draw_xs = np.rint(draw_xs).astype(int)
        draw_ys = np.rint(draw_ys).astype(int)
        in_bounds = ((0 <= draw_xs) & (draw_xs < im.shape[1])
                     & (0 <= draw_ys) & (draw_ys < im.shape[0]))
        draw_xs = draw_xs[in_bounds]
        draw_ys = draw_ys[in_bounds]
        im[draw_ys, draw_xs, 0] = True
        P.imshow(im)
        P.show()
    self.notehead_model = np.median(self.model_ellipses, axis=0)
    self.page.notehead_model = self.notehead_model
    print self.notehead_model

  def create_ellipse_model_mask(self):
    # Create a bitmap of the ellipse centered at (a, a)
    # Then find the normal at each point on the bitmap
    x0, y0, a, b, t = self.notehead_model
    aval = np.ceil(a).astype(int) + 1
    self.ellipse_mask = np.zeros((2*aval+1, 2*aval+1), dtype=bool)
    ts = np.linspace(0, 2*np.pi, 1000)
    ellipse_xs = aval + (a+1)*np.cos(ts)*np.cos(t) - (b+1)*np.sin(ts)*np.sin(t)
    ellipse_ys = aval + (a+1)*np.cos(ts)*np.sin(t) + (b+1)*np.sin(ts)*np.cos(t)
    ellipse_xs = np.rint(ellipse_xs).astype(int)
    ellipse_ys = np.rint(ellipse_ys).astype(int)
    self.ellipse_mask[ellipse_ys, ellipse_xs] = True

    ts_normal = ts - t # Rotate ellipse in -t direction
    # Find angle of normal vector with non-rotated ellipse and add t
    normal_angle = np.arctan2(b * np.cos(ts_normal),
                              a * np.sin(ts_normal)) + t
    self.ellipse_normals = np.zeros_like(self.ellipse_mask, dtype=np.double)
    self.ellipse_normals[ellipse_ys, ellipse_xs] = normal_angle

    # Unique coordinates of ellipse mask
    self.ellipse_mask_y, self.ellipse_mask_x = np.where(self.ellipse_mask)
    self.ellipse_mask_y = self.ellipse_mask_y - aval
    self.ellipse_mask_x = self.ellipse_mask_x - aval

  def search_glyph(self, g):
    x0, y0, a, b, t = self.notehead_model
    a = np.ceil(a).astype(int)
    glyph_box = self.page.glyph_boxes[g]
    glyph = (self.page.labels[glyph_box] == g+1).astype(int)
    glyph_border = glyph & (ndimage.convolve(glyph,
                                             [[1,1,1], [1,0,1], [1,1,1]],
                                             mode='constant') < 8)
    # Consider any point inside the glyph (e.g. space in half note)
    ndimage.binary_closing(glyph, iterations=3, output=glyph)
    ndimage.binary_fill_holes(glyph, output=glyph)
    glyph_gradient = self.page.gradient[(0,) + glyph_box]
    glyph_gradient_magnitude = self.page.gradient[(1,) + glyph_box]
    candidate_ys, candidate_xs = np.where(glyph)
    candidate_scores = np.empty_like(candidate_ys, dtype=np.double)
    i=0
    for y, x in zip(candidate_ys, candidate_xs):
      mask_ys = y + self.ellipse_mask_y
      mask_xs = x + self.ellipse_mask_x
      mask_points = np.ones_like(mask_ys, dtype=bool)
      mask_points &= 0 <= mask_ys
      mask_points &= mask_ys < glyph.shape[0]
      mask_points &= 0 <= mask_xs
      mask_points &= mask_xs < glyph.shape[1]
      candidate_scores[i] = np.count_nonzero(glyph_border[mask_ys[mask_points], mask_xs[mask_points]])
      i += 1
    winner = np.argmax(candidate_scores)
    win = np.amax(candidate_scores)
    import pylab as P
    im = np.zeros(glyph.shape + (3,), dtype=np.uint8)
    im[..., 2] = glyph * 255
    im[candidate_ys, candidate_xs, 0] = np.rint(candidate_scores * 255 / win)
    P.imshow(im); P.show()
    return candidate_ys[winner], candidate_xs[winner]
  def process(self):
    self.choose_model_glyph_candidates()
    self.choose_model_glyphs()
    self.fit_model_ellipses()
    self.create_ellipse_model_mask()
    #print self.search_glyph(316)

  def color_image(self):
    pass
