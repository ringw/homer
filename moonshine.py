import image
import staff
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--colored", type=str, help="Highlighted image output")
parser.add_argument("path", type=str, help="path to scanned music")

def moonshine(path, colored=None):
  page, = image.read_pages(path)
  task = staff.StaffTask(page)
  task.process()
  if colored:
    task.color_image()
    page.colored.save(colored)

if __name__ == "__main__":
  args = parser.parse_args()
  moonshine(args.path, colored=args.colored)
