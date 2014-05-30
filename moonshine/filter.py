from .opencl import *
from . import bitimage

prg = build_program("filter")
prg.staff_removal_filter.set_scalar_arg_dtypes([
    None, np.uint32, np.uint32, None,
])
prg.barline_filter.set_scalar_arg_dtypes([
    None, np.int32, None,
])

def remove_staff(page, img=None):
    if img is None:
        img = page.img
    output = cla.zeros_like(img)
    prg.staff_removal_filter(q, img.shape[::-1], (16, 16),
                                img.data,
                                np.uint32(page.staff_thick+1),
                                np.uint32(page.staff_dist),
                                output.data).wait()
    return output

def barline_filter(page, img=None):
    if img is None:
        img = page.img
    no_staff = remove_staff(page, img)
    no_staff_T = bitimage.transpose(no_staff)
    output = cla.zeros_like(no_staff_T)
    prg.barline_filter(q, (img.shape[0] // 8, img.shape[1] * 8), # W,H
                          (16, 16),
                          no_staff_T.data,
                          np.int32(page.staff_thick+1),
                          output.data).wait()
    return bitimage.transpose(output)
