import sys
import image
import staff
import numpy as np

def moonshine(path):
  pages = image.open_file(path)
  task = staff.StaffTask(pages[0])
  task.process()
  p = pages[0]

if __name__ == "__main__":
  moonshine(sys.argv[1])
