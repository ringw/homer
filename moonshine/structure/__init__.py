from . import orientation, staffsize, barlines, systems, staffboundary

def process(page):
    staffsize.staffsize(page)
    orientation.rotate(page)
    # If we rotate significantly, the vertical difference between staff lines
    # may be slightly different
    staffsize.staffsize(page)

    page.staves()
    barlines.get_barlines(page)
    systems.build_systems(page)
    staffboundary.boundaries(page)
