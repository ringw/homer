"""
Measure fingerprinting

We extract one pixel from each staff space in the original image.
The run-length encoding of each measure is used to create a unique fingerprint,
and scores can be compared by the fingerprints of each measure to compute
similarity.
"""
import numpy as np

def staff_extract_spaces(page, staff_num):
    staff = page.staves.get_staff(staff_num)
    xmin = np.amin(staff[:, 0])
    xmax = np.amax(staff[:, 0]) + 1
    sd = page.staves.staff_dist[staff_num]
    line_space = sd // 2 - max(2, page.staff_thick)

    space_list = []
    for space in [-5, -3, -1, 1, 3, 5]:
        space_pixels = np.empty((line_space*2 + 1, xmax - xmin), bool)
        for x in xrange(xmin, xmax):
            y_center = page.staves.staff_y(staff_num, x)
            y_space = y_center + (space * sd) // 2
            space_pixels[:, x - xmin] = page.byteimg[y_space - line_space
                                                : y_space + line_space + 1, x]
        # Must have two consecutive black pixels
        space_row = (space_pixels[:-1, :] & space_pixels[1:, :]).any(axis=0)
        space_list.append(space_row)

    spaces = np.vstack(space_list)
    return spaces
