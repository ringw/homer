import numpy as np
from scipy.ndimage import label

class Glyph:
  (EXTRANEOUS, # glyphs we can ignore
   NOISE, # small unidentifiable noise
   BLOB, # formed from multiple valid glyphs
   CHUNK, # unidentifiable piece of a glyph
   TEXT,
   BARLINE,
   NOTEHEAD_FULL,
   NOTEHEAD_EMPTY,
    ) = xrange(8)

class GlyphsTask:
  def __init__(self, page):
    self.page = page

  def label_initial_glyphs(self):
    self.labelled_glyphs, self.num_glyphs = label(self.page.im, np.ones((3,3)))

  # Source: http://en.wikipedia.org/wiki/Image_moment
  def get_hu_moments(self):
    labels = self.labelled_glyphs.ravel().astype(int)
    # Each row is x^0, x^1, ... x^3
    x_series = np.vander(np.arange(0, self.page.im.shape[1]), 4)[:, -1::-1].T
    y_series = np.vander(np.arange(0, self.page.im.shape[0]), 4)[:, -1::-1].T
    # Tile series to be used as weights for moments
    x_weights = np.tile(x_series, self.page.im.shape[0])
    y_weights = np.repeat(y_series, self.page.im.shape[1], axis=1)
    # Raw moments (only calculate the ones we use)
    M = np.zeros((4, 4, self.num_glyphs))
    # XXX: Can we speed this up by counting all at once?
    for i in range(0,3):
      for j in range(0,3):
        if i == 2 and j == 2: continue
        M[i,j] = np.bincount(labels, weights=x_weights[i]*y_weights[j])[1:]
    xbar = M[1,0]/M[0,0]
    ybar = M[0,1]/M[0,0]
    M[3,0] = np.bincount(labels, weights=x_weights[3])[1:]
    M[0,3] = np.bincount(labels, weights=y_weights[3])[1:]
    # Central moments
    mu = np.zeros((4, 4, self.num_glyphs))
    mu[0,0] = M[0,0]
    mu[1,1] = M[1,1] - xbar*M[0,1]
    mu[2,0] = M[2,0] - xbar*M[1,0]
    mu[0,2] = M[0,2] - ybar*M[0,1]
    mu[2,1] = M[2,1] - 2*xbar*M[1,1] - ybar*M[2,0] + 2 * xbar**2 * M[0,1]
    mu[1,2] = M[1,2] - 2*ybar*M[1,1] - xbar*M[0,2] + 2 * ybar**2 * M[1,0]
    mu[3,0] = M[3,0] - 3*xbar*M[2,0] + 2 * xbar**2 * M[1,0]
    mu[0,3] = M[0,3] - 3*ybar*M[0,2] + 2 * ybar**2 * M[0,1]
    # Scale invariant moments
    eta = np.zeros((4, 4, self.num_glyphs))
    for i in range(0,3):
      for j in range(0,3):
        if i == 2 and j == 2: continue
        eta[i,j] = mu[i,j]/np.power(mu[0,0], 1 + float(i+j)/2)
    eta[3,0] = mu[3,0] / np.power(mu[0,0], 2.5)
    eta[0,3] = mu[0,3] / np.power(mu[0,0], 2.5)
    # Rotation invariant moments
    I = np.zeros((self.num_glyphs, 8))
    I[:,0] = eta[2,0] + eta[0,2]
    I[:,1] = (eta[2,0] - eta[0,2])**2 + 4*eta[1,1]**2
    I[:,2] = (eta[3,0] - 3*eta[1,2])**2 + (3*eta[2,1] - eta[0,3])**2
    I[:,3] = (eta[3,0] + eta[1,2])**2 + (eta[2,1] + eta[0,3])**2
    I[:,4] = (eta[3,0] - 3*eta[1,2])*(eta[3,0] + eta[1,2]) \
               * ((eta[3,0] + eta[1,2])**2 - 3*(eta[2,1] + eta[0,3])**2) \
           + (3*eta[2,1] - eta[0,3])*(eta[2,1] + eta[0,3]) \
               * (3*(eta[3,0] + eta[1,2])**2 - (eta[2,1] + eta[0,3])**2)
    I[:,5] = (eta[2,0] - eta[0,2]) \
               * ((eta[3,0] + eta[1,2])**2 - (eta[2,1] + eta[0,3])**2) \
           + 4*eta[1,1]*(eta[3,0] + eta[1,2])*(eta[2,1] + eta[0,3])
    I[:,6] = (3*eta[2,1] - eta[0,3])*(eta[3,0] + eta[1,2]) \
               * ((eta[3,0]+eta[1,2])**2 - 3*(eta[2,1]+eta[0,3]**2)) \
           - (eta[3,0] - 3*eta[1,2])*(eta[2,1] + eta[0,3]) \
               * (3*(eta[3,0] + eta[1,2]**2 - (eta[2,1] + eta[0,3])**2))
    # Eighth third order independent moment invariant (from Wikipedia)
    I[:,7] = eta[1,1] * ((eta[3,0]+eta[1,2])**2 - (eta[0,3]+eta[2,1])**2) \
             - (eta[2,0]-eta[0,2])*(eta[3,0]+eta[1,2])*(eta[0,3]+eta[2,1])
    self.hu_moments = I

  def process(self):
    self.label_initial_glyphs()
    self.get_hu_moments()

  def color_image(self):
    pass
