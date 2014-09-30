# Scale page to form superpixels, then run an HMM on each staff
from .. import bitimage

def scaled_staff(page, staff_num):
    """ Scale each staff so that staff_dist ~= 4, and scale horizontally
        by a factor of 1 / max(1, staff_thick/2). """
    scale_x = 1.0 / max(1, (page.staff_thick + 1) // 2)
    scale_y = 5.0 / float(page.staff_dist)
    extracted = page.staves.extract_staff(staff_num, page.staves.nostaff())
    scaled_img = bitimage.scale(extracted, scale_x, scale_y)
    return scaled_img[:24]
