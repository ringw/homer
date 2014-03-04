from . import debug, image, page
import numpy as np
import argparse

def open(image_path):
  images = image.read_pages(image_path)
  return map(page.Page, images)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", type=str,
                    help="Comma-separated modules to debug")
parser.add_argument("-p", "--page", type=int)
parser.add_argument("-s", "--show", dest="show", action="store_true")
parser.add_argument("-S", "--no-show", dest="show", action="store_false")
parser.set_defaults(show=True)
parser.add_argument("path", type=str, help="path to scanned music")

def moonshine(path, page=0, show=False):
  page = open(path)[page]
  page.process()
  if show:
    import pylab
    page.show()
    pylab.show()
