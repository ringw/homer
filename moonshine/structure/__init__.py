from . import staves, staffsystems, staffboundary

def process(page):
    staves.staves(page)
    staffsystems.staff_systems(page)
    staffboundary.boundaries(page)
