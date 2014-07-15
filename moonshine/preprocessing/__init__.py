from . import orientation, staffsize

def process(page):
    staffsize.staffsize(page)
    orientation.rotate(page)
    # If we rotate significantly, the vertical difference between staff lines
    # may be slightly different
    staffsize.staffsize(page)
