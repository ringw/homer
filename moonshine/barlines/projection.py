# Detect barlines for each single staff from the horizontal projection of the
# slice of the image containing the staff.
# Next, barlines close to each other on adjacent staves need to be checked
# to see if they are joined, in which case we join the staves into one system.
from ..gpu import *
from .. import bitimage, filter, util

def staff_barlines(page, staff_num):
    img_slice = bitimage.as_hostimage(page.staves.extract_staff(staff_num,
                                                page.barline_filter,
                                                extract_lines=8))
    # The barlines should contain mostly black pixels just in the actual staff,
    # but we need to check above and below the staff for other symbols
    staff_proj = img_slice[page.staff_dist*2:page.staff_dist*6, :].sum(0)
    gap_proj = img_slice.sum(0)

    # Barline must take up at least 80% of the vertical space,
    # and there should be background (few black pixels) around it
    is_barline = staff_proj > page.staff_dist * 4 * 0.8
    is_background = gap_proj <= page.staff_dist
    is_in_barline = util.closing_1d(is_barline, page.staff_thick)
    labels, num_labels = util.label_1d(is_in_barline)
    if not labels.any():
        return np.array([], int)
    barlines = []
    for label in xrange(1, num_labels+1):
        pos, = np.where(labels == label)
        if pos[-1] - pos[0] > page.staff_dist * 1.5:
            continue
        left_bg = is_background[max(0, pos[0] - page.staff_dist/2) : pos[0]]
        right_bg = is_background[pos[-1] : min(len(gap_proj),
                                               pos[-1] + page.staff_dist/2)]
        if left_bg.sum() >= len(left_bg)/2 and right_bg.sum() >=len(right_bg)/2:
            barlines.append([pos[0], pos[-1]])
    barlines = np.array(barlines, int)

    # Add a barline at the start and end of the staff if necessary
    if len(barlines):
        staff = page.staves()[staff_num]
        if barlines[0, 0] - staff[0,0] > page.staff_dist*2:
            barlines = np.concatenate([[[staff[0,0], staff[0,0]]], barlines])
        if staff[1,0] - barlines[-1, 1] > page.staff_dist*2:
            barlines = np.concatenate([barlines, [[staff[1,0], staff[1,0]]]])
    return barlines

def get_barlines(page):
    page.barline_filter = page.staves.nostaff()
    page.barlines = [staff_barlines(page, i)
                     for i in xrange(len(page.staves()))]
    del page.barline_filter
    return page.barlines

def show_barlines(page):
    import pylab
    for i, barlines in enumerate(page.barlines):
        for j, barline_range in enumerate(barlines):
            barline_x = int(barline_range.mean())
            staff_y = page.staves.staff_y(i, barline_x)
            repeats = page.repeats[i][j]
            if repeats:
                # Draw thick bar
                pylab.fill_between([barline_x - page.staff_dist/4,
                                    barline_x + page.staff_dist/4],
                                   staff_y - page.staff_dist*2,
                                   staff_y + page.staff_dist*2,
                                   color='g')
                for letter, sign in (('L', -1), ('R', +1)):
                    if letter in repeats:
                        # Draw thin bar
                        bar_x = barline_x + sign * page.staff_dist/2
                        pylab.plot([bar_x, bar_x],
                                   [staff_y - page.staff_dist*2,
                                    staff_y + page.staff_dist*2],
                                   color='g')
                        for y in (-1, +1):
                            circ = pylab.Circle((bar_x + sign*page.staff_dist/2,
                                                 staff_y + y*page.staff_dist/2),
                                                page.staff_dist/4,
                                                color='g')
                            pylab.gcf().gca().add_artist(circ)
            else:
                pylab.plot([barline_x, barline_x],
                           [staff_y - page.staff_dist*2,
                            staff_y + page.staff_dist*2],
                           color='g')
