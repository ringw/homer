# Detect barlines for each single staff from the horizontal projection of the
# slice of the image containing the staff.
# Next, barlines close to each other on adjacent staves need to be checked
# to see if they are joined, in which case we join the staves into one part.
from .opencl import *
from . import bitimage, filter, util

def staff_barlines(page, staff_num):
    staff_y = int(page.staves[staff_num, [2,3]].sum()/2.0)
    y0 = staff_y - page.staff_dist * 2
    y1 = staff_y + page.staff_dist * 2
    img = filter.remove_staff(page)
    img_slice = bitimage.as_hostimage(img[y0:y1, :])
    proj = img_slice.sum(0)

    # Barline must take up at least 75% of the vertical space,
    # and there should be background (few active pixels) around it
    is_barline = proj > (y1 - y0) * 0.9
    is_background = proj < page.staff_dist/2
    near_background_left = is_background.copy()
    near_background_right = is_background.copy()
    for i in range(page.staff_thick, page.staff_dist):
        near_background_left[i:] |= is_background[:-i]
    for i in range(page.staff_thick, page.staff_dist/2):
        near_background_right[:-i] |= is_background[i:]
    is_barline &= near_background_left & near_background_right
    #import pylab
    #pylab.subplot(211)
    #pylab.plot(is_barline)
    #pylab.xlim([0,4096])
    #pylab.subplot(212)
    #pylab.imshow(img_slice)
    from moonshine import util
    labels, num_labels = util.label_1d(is_barline)
    barlines = np.rint(util.center_of_mass_1d(labels)).astype(int)
    return barlines
