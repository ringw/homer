class Page:
  def __init__(self, im, colored=None):
    self.im = im
    self.colored = colored # RGB copy of image for coloring
    self.staves = []
