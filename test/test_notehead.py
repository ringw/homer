import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import moonshine
from moonshine import measure, notehead
page, = moonshine.open('samples/sonata.png')
page.process()

m = measure.get_measure(page, 11, 3)
e, pairs, ell = notehead.detect_ellipses(page, m)
e.wait()
print (e.profile.end - e.profile.start) / 10.0**9
import scipy.stats
from pylab import *
E = ell.get()
axis = scipy.stats.mode(E[E != 0])[0][0]
print "axis:", axis
imshow(np.unpackbits(m.get()).reshape((m.shape[0],-1)))
for x0,y0,x1,y1 in pairs.get()[E == axis].astype(int):
    xc = (x0 + x1)/2.0
    yc = (y0 + y1)/2.0
    a = sqrt((x1 - x0)**2 + (y1 - y0)**2)/2
    t0 = -np.arctan2(y1 - y0, x1 - x0)
    ts = np.linspace(0, 2*pi, 100)
    e_x = a * cos(ts)
    e_y = axis * sin(ts)
    plot(xc + e_x * cos(t0) + e_y * -sin(t0),
         yc + e_y * cos(t0) + e_x * sin(t0), 'g')
show()
