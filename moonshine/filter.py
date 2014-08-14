from .gpu import *
from . import bitimage

prg = build_program("filter")

LOCAL_SIZE = (16, 16)

def staff_center(page, img=None):
    if img is None:
        img = page.img
    output = thr.empty_like(img)
    prg.staff_center_filter(img,
                            np.int32(page.staff_dist),
                            output,
                            global_size=img.shape[::-1])
    return output

def remove_staves(page, img=None):
    if img is None:
        img = page.img
    output = thr.empty_like(img)
    prg.staff_removal_filter(img,
                             np.int32(page.staff_thick+1),
                             np.int32(page.staff_dist),
                             output,
                             global_size=img.shape[::-1])
    return output

def barline_filter(page, img=None):
    if img is None:
        img = page.img
    no_staff = remove_staff(page, img)
    no_staff_T = bitimage.transpose(no_staff)
    output = cla.zeros_like(no_staff_T)
    prg.barline_filter(no_staff_T,
                       np.int32(page.staff_thick+1),
                       output,
                       global_size=(img.shape[0] // 8, img.shape[1] * 8))
    return bitimage.transpose(output)
