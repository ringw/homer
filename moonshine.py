import numpy as np
import image
import staff
import vertical
import glyph
import gradient
import argparse
import notehead

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--colored", type=str, help="Highlighted image output")
parser.add_argument("path", type=str, help="path to scanned music")

def moonshine(path, colored=None):
  page, = image.read_pages(path)
  tasks = [staff.StaffTask(page),
           gradient.GradientTask(page),
           vertical.VerticalsTask(page),
           glyph.GlyphsTask(page),
           notehead.NoteheadsTask(page)]
  for task in tasks:
    task.process()
  if colored:
    for task in tasks:
      task.color_image()
    page.colored.save(colored)

if __name__ == "__main__":
  args = parser.parse_args()
  moonshine(args.path, colored=args.colored)
