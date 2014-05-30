from . import rotate, staffsize

def process(page):
    rotate.rotate(page)
    staffsize.staffsize(page)
