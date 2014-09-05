from . import hough

# hough is default implementation
staves = hough.staves

def show_staves(page):
    import pylab as p
    for (x0, y0), (x1, y1) in page.staves:
        p.plot([x0, x1], [y0, y1], 'g')
