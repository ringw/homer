# Join adjacent staves into a system if their barlines line up.
from .. import opencl

def initialize_systems(page):
    # Initial systems are each individual staff,
    # barlines are a vertical line through the staff
    page.systems = []
    for staff, barlines in zip(page.staves, page.barlines):
        x0, x1, y0, y1 = staff
        system_bars = []
        for barline_x in barlines:
            staff_y = y0 + (y1 - y0) * (barline_x - x0) / (x1 - x0)
            system_bars.append([barline_x, barline_x,
                                staff_y - page.staff_dist*2,
                                staff_y + page.staff_dist*2])
        page.systems.append(system_bars)
