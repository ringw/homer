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

  def process(self):
    g = 110
    # 100, 106, 110
    glyph_y, glyph_x = self.page.glyph_boxes[g]
    glyph_border = (self.page.labels[glyph_y, glyph_x] == (g+1)).astype(int)
    glyph_border &= (convolve2d(glyph_border, [[1,1,1],[1,0,1],[1,1,1]],
                                mode='same') < 8)
    y,x = ellipse_candidate_centers(glyph_border, self.page.gradient[0, glyph_y, glyph_x], radius_max=self.page.staff_space)
    print (y+glyph_y.start, x+glyph_x.start)

  def color_image(self):
    pass
