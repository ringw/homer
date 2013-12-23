import numpy as np
import debug
import image
import argparse

open = image.read_pages

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", type=str,
                    help="Comma-separated modules to debug")
parser.add_argument("-s", "--show", dest="show", action="store_true")
parser.add_argument("-S", "--no-show", dest="show", action="store_false")
parser.set_defaults(show=True)
parser.add_argument("path", type=str, help="path to scanned music")

def moonshine(path, show=False):
  page = open(path)[0]
  page.process()
  if show:
    import pylab
    page.show()
    pylab.show()

if __name__ == "__main__":
  args = parser.parse_args()
  if args.debug:
    debug.DEBUG_MODULES = args.debug.split(',')
  moonshine(args.path, show=args.show)
