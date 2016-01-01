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
    return spaces, xmin, xmax

def dilate_1d(img, niter):
    result = img.copy()
    result[:, :-1] |= img[:, 1:]
    result[:, 1:]  |= img[:, :-1]
    return result
def erode_1d(img, niter):
    result = img.copy()
    result[:, :-1] &= img[:, 1:]
    result[:, 1:]  &= img[:, :-1]
    return result
def open_close(img, niter):
    # Equivalent to binary opening followed by closing
    return erode_1d(dilate_1d(erode_1d(img, niter),
                              2*niter), niter)

def spaces_to_bytes(spaces, niter=2):
    assert spaces.shape[0] == 6
    spaces = open_close(spaces, niter)
    result = np.zeros(spaces.shape[1], np.uint8)
    result |= np.where(spaces[0,:], 0x20, 0)
    result |= np.where(spaces[1,:], 0x10, 0)
    result |= np.where(spaces[2,:], 0x08, 0)
    result |= np.where(spaces[3,:], 0x04, 0)
    result |= np.where(spaces[4,:], 0x02, 0)
    result |= np.where(spaces[5,:], 0x01, 0)
    return result

def bytes_to_fingerprint(cols, skip=2, short_cutoff=8):
    "skip and short_cutoff should be determined based on staff_dist"
    runs = []
    for col in cols:
        # If end of run, delete the last run if we could skip it
        if runs and runs[-1]['value'] != col and runs[-1]['length'] <= skip:
            del runs[-1]

        if runs and runs[-1]['value'] == col:
            runs[-1]['length'] += 1
        else:
            runs.append(dict(value=col, length=1))

    if runs and runs[-1]['length'] <= skip:
        del runs[-1]

    fingerprint = ""
    for run in runs:
        fingerprint += chr(run['value'] | 0x40
                           if run['length'] >= short_cutoff
                           else run['value'])
    return fingerprint

def fingerprint_measure(page, measure_spaces):
    niter = -(-page.staff_thick // 2)
    spaces_bytes = spaces_to_bytes(measure_spaces, niter)

    skip = page.staff_thick
    short_cutoff = page.staff_dist // 2
    fingerprint = bytes_to_fingerprint(spaces_bytes, skip, short_cutoff)

    return fingerprint
