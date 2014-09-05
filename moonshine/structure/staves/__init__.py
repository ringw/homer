from . import hough, path

# hough is default implementation
staves = path.staves

def show_staves(page):
    import pylab as p
    # Plot each segment of the staff center line
    for staff in page.staves:
        xs = staff[:, 0]
        ys = staff[:, 1]
        p.plot(xs, ys, 'g')
