import numpy as np
import scipy.ndimage.morphology as m
import skimage.morphology
import ImageDraw
import matplotlib.pyplot as plt

class LinesTask:
  def __init__(self, page):
    self.page = page
  def process(self):
    g = 100
    glyph_y, glyph_x = self.page.glyph_boxes[g]
    #glyph = (self.page.labels[glyph_y, glyph_x] == (g+1))
    #glyph_border &= (convolve2d(glyph_border, [[1,1,1],[1,0,1],[1,1,1]],
    #                            mode='same') < 8)
    #print 'skeletonize...'
    skel = skimage.morphology.medial_axis(self.page.im)
    #print 'done'
    g = skel[glyph_y, glyph_x]
    plot = np.zeros_like(g, dtype=int)
    gy,gx = np.where(g)
    plot[gy,gx] = self.page.gradient[0,gy+glyph_y.start, gx+glyph_x.start]
    #plt.clf()
    #plt.imshow(plot)
    #plt.colorbar()
    #plt.show()
  def color_image(self):
    d = ImageDraw.Draw(self.page.colored)
    #for barline in self.page.barlines:
    #  d.line(tuple(np.rint(barline[[2, 0, 3, 1]]).astype(int)), fill=(0,0,255))
