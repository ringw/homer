import numpy as np
import debug
import image
import argparse

open = image.read_pages

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", type=str,
                    help="Comma-separated modules to debug")
parser.add_argument("path", type=str, help="path to scanned music")

def moonshine(path, colored=None):
  page, = open(path)
  page.process()

if __name__ == "__main__":
  args = parser.parse_args()
  if args.debug:
    debug.DEBUG_MODULES = args.debug.split(',')
  moonshine(args.path, colored=args.colored)
