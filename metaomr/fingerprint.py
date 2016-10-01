"""
Measure fingerprinting

We extract one pixel from each staff space in the original image.
The run-length encoding of each measure is used to create a unique fingerprint,
and scores can be compared by the fingerprints of each measure to compute
similarity.
"""
from cStringIO import StringIO
import hashlib
import numpy as np
import struct

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

def fingerprint_page(page, grid_size=16):
    "Extract scale-invariant fingerprint from the systems on the page"
    # Find the average angle of each system to undo rotation.
    # This may be a useful step to undo rotation on the actual image.
    # Also, fit a quadratic for deskewing?
    angles = []
    for sys in page.systems:
        xs = sys['barlines'][:, :, 0].mean(1)
        ys = sys['barlines'][:, :, 1].mean(1)
        coefs = np.polyfit(xs, ys, 1)
        theta = np.arctan(coefs[0])
        angles.append(theta)
    theta = np.mean(angles)
    # Rotate all x values by -theta
    rotation = np.array([[np.cos(-theta), -np.sin(-theta)],
                         [np.sin(-theta),  np.cos(-theta)]])
    systems_xs = []
    for sys in page.systems:
        def barline_rotated_x(barline):
            center = barline.mean(0)
            rotated_center = rotation.dot(center[:, None])
            return rotated_center[0, 0]
        systems_xs.append(map(barline_rotated_x, sys['barlines']))
    # Make margin-invariant
    xmin = min(min(xs) for xs in systems_xs)
    xmax = max(max(xs) for xs in systems_xs)
    width = xmax - xmin
    systems_xs = [[x - xmin for x in xs] for xs in systems_xs]
    systems_str = StringIO()
    for xs in systems_xs:
        grid_pos = ((np.asarray(xs) * grid_size) / width).astype(int)
        grid_pos[grid_pos >= grid_size] = grid_size - 1
        systems_str.write(','.join(map(str, grid_pos)))
        systems_str.write('\n')
    systems_str.seek(0)
    systems_str = systems_str.read()
    sha = hashlib.sha1(systems_str).digest()
    fingerprint, = struct.unpack("<L", sha[:4])
    return fingerprint
