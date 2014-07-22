from .. import preprocessing
from . import staves, barlines, systems, staffboundary

def process(page):
    if not hasattr(page, 'staff_dist'):
        preprocessing.process(page)
    staves.staves(page)
    barlines.get_barlines(page)
    systems.build_systems(page)
    staffboundary.boundaries(page)
