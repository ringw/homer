from . import staves, barlines, systems, staffboundary

def process(page):
    staves.staves(page)
    barlines.get_barlines(page)
    systems.build_systems(page)
    staffboundary.boundaries(page)
