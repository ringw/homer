from .gpu import *
import numpy as np
from fractions import gcd

int2 = np.dtype('i4,i4')
int4 = np.dtype('i4,i4,i4,i4')
prg = build_program("copy_measure")
def get_measure(page, staff, measure):
    for system in page.systems:
        barlines = system['barlines']
        bar_start, bar_stop = system['start'], system['stop']
        if bar_start <= staff and staff <= bar_stop:
            break
    else:
        raise Exception("Staff not found in barlines")
    if measure + 1 >= len(barlines):
        raise Exception("Measure out of bounds")
    # Round down measure start x
    x0 = barlines[measure, [0,1]].min() & -8
    # Round up measure end x
    x1 = -(-barlines[measure + 1, [0,1]].max() & -8)

    # Round up staff start y
    y0 = page.boundaries[staff][:, 1].min() & -8
    # Round down staff end y
    y1 = -(-page.boundaries[staff+1][:, 1].max() & -8)

    measure_pixel_size = (y1 - y0, (x1 - x0) // 8)
    measure_size = tuple(-(-i & -16) for i in measure_pixel_size)
    measure = cla.zeros(q, measure_size, np.uint8)
    device_b0 = cla.to_device(q, page.boundaries[staff][:, 1].astype(np.int32))
    device_b1 = cla.to_device(q, page.boundaries[staff + 1][:, 1]
                                     .astype(np.int32))
    prg.copy_measure(q, measure.shape[::-1], (1, 1),
                        page.img.data,
                        np.array(page.img.shape[::-1],
                                 np.int32).view(int2)[0],
                        device_b0.data,
                        np.int32(page.boundaries[staff][1, 0]
                                    - page.boundaries[staff][0, 0]),
                        device_b1.data,
                        np.int32(page.boundaries[staff+1][1, 0]
                                    - page.boundaries[staff+1][0, 0]),
                        measure.data,
                        np.array([x0 // 8, y0] + list(measure_size[::-1]),
                                 np.int32).view(int4)[0]).wait()
    return measure, (x0, x1, y0, y1)

class Measure:
    page = None
    staff_num = None
    measure_num = None
    image = None
    bounds = None
    start_pitch = None # Set to clef and key signature of previous measure
    pitch_elements = None
    final_pitch = None # Final clef and key signature after this measure
    def __init__(self, page, staff_num, measure_num):
        self.page = page
        self.staff_num = staff_num
        self.measure_num = measure_num
        self.page_staff_y = page.staves[staff_num, [2,3]].sum()/2.0

    def get_image(self):
        if self.image is None:
            self.image, self.bounds = get_measure(self.page,
                                            self.staff_num, self.measure_num)
            self.staff_y = self.page_staff_y - self.bounds[2]
        return self.image

    def show_elements(self, x0=0, y0=0, on_page=False):
        if not hasattr(self, 'elements'):
            return
        # on_page overrides x0 and y0 to draw in page coordinates
        if on_page:
            x0 = self.bounds[0]
            y0 = self.bounds[2]
        import pylab
        for x, vert_slice in self.elements:
            for label, y in vert_slice:
                staff_y0 = y0 + self.page_staff_y - self.bounds[2]
                pylab.plot(x0 + x * self.page.staff_dist / 8.0,
                           staff_y0 - self.page.staff_dist * y / 2.0, 'r.')
def build_bars(page):
    bars = []
    for system in page.systems:
        bar = []
        for measure in xrange(len(system['barlines']) - 1):
            m = []
            for staff in xrange(system['start'], system['stop']+1):
                m.append(Measure(page, staff, measure))
            bar.append(m)
        bars.append(bar)
    page.bars = bars
