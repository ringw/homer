import numpy as np
import debug
import image
import staff
import line
import glyph
import gradient
import argparse
import notehead

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--colored", type=str, help="Highlighted image output")
parser.add_argument("-d", "--debug", type=str, help="Comma-separated modules to debug")
parser.add_argument("path", type=str, help="path to scanned music")

def moonshine(path, colored=None):
  page, = image.read_pages(path)
  tasks = [staff.StavesTask(page),
           gradient.GradientTask(page),
           glyph.GlyphsTask(page),
           line.LinesTask(page),
           notehead.NoteheadsTask(page)]
  for task in tasks:
    task.process()
  if colored:
    for task in tasks:
      task.color_image()
    page.colored.save(colored)

if __name__ == "__main__":
  args = parser.parse_args()
  if args.debug:
    debug.DEBUG_MODULES = args.debug.split(',')
  moonshine(args.path, colored=args.colored)
