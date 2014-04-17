import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import moonshine
from moonshine import measure, notehead
page, = moonshine.open('samples/sonata.png')
page.process()

from pylab import *
for i, bar in enumerate(page.barlines):
    for staff in xrange(bar[0], bar[1]):
        for j in xrange(len(bar[2]) - 1):
            m = measure.get_measure(page, staff, j)
            figure()
            e, pairs, ell = notehead.detect_ellipses(page, m)
            e.wait()
            print (e.profile.end - e.profile.start) / 10.0**9
            import scipy.stats
            E = ell.get()
            axis = scipy.stats.mode(E[E != 0])[0][0]
            print "axis:", axis
            print "% detected", 100 * float((E != 0).sum()) / E.shape[0]
            figure()
            imshow(np.unpackbits(m.get()).reshape((m.shape[0],-1)))
            P = pairs.get().view(np.int32)
            for x0,y0,x1,y1 in P:
                if x1 >= 0:
                    plot([x0,x1], [y0,y1], 'y.')
            for x0,y0,x1,y1 in P[E > 0].astype(int):
                xc = (x0 + x1)/2.0
                yc = (y0 + y1)/2.0
                a = sqrt((x1 - x0)**2 + (y1 - y0)**2)/2
                t0 = np.arctan2(y1 - y0, x1 - x0)
                ts = np.linspace(0, 2*pi, 100)
                e_x = a * cos(ts)
                e_y = axis * sin(ts)
                plot(xc + e_x * cos(t0) - e_y * sin(t0),
                     yc + e_y * cos(t0) + e_x * sin(t0),
                     'g')
            break
#figure()
#img = np.zeros((m.shape[0], m.shape[1] * 8), int)
#img[np.rint(yc + e_y * cos(t0) + e_x * sin(t0)).astype(int), np.rint(xc + e_x * cos(t0) + e_y * -sin(t0)).astype(int)] = 1
#bits = np.packbits(img)
#from moonshine.opencl import *
#img_device = cla.to_device(q, bits.reshape((m.shape[0], -1)))
#e, pairs, ell = notehead.detect_ellipses(page, img_device)
#E = ell.get()
#if (E != 0).any():
#    axis = scipy.stats.mode(E[E != 0])[0][0]
#    print "axis:", axis
#    figure()
#    imshow(img)
#    for x0,y0,x1,y1 in pairs.get()[E == axis].astype(int):
#        plot([x0, x1], [y0, y1], 'y.')
#        xc = (x0 + x1)/2.0
#        yc = (y0 + y1)/2.0
#        a = sqrt((x1 - x0)**2 + (y1 - y0)**2)/2
#        t0 = -np.arctan2(y1 - y0, x1 - x0)
#        ts = np.linspace(0, 2*pi, 100)
#        e_x = a * cos(ts)
#        e_y = axis * sin(ts)
#        plot(xc + e_x * cos(t0) + e_y * -sin(t0),
#             yc + e_y * cos(t0) + e_x * sin(t0), 'g')
#show()
