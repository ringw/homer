from .opencl import *
from . import bitimage

prg = cl.Program(cx, """
__kernel void staff_removal_filter(__global const uchar *image,
                                   uint staff_thick,
                                   uint staff_dist,
                                   __global uchar *output_image) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint w = get_global_size(0);
    uint h = get_global_size(1);

    uchar byte = image[x + w * y];
    uchar is_staff = byte;
    // Expect empty space above or below
    if (0 <= y - staff_thick && y + staff_thick < h) {
        is_staff &= ~ image[x + w * (y - staff_thick)];
        is_staff &= ~ image[x + w * (y + staff_thick)];
    }
    // Expect another staff either above or below
    uchar staff_above_below = 0;
    if (0 <= y - staff_dist)
        staff_above_below |= image[x + w * (y - staff_dist)];
    if (y + staff_dist < h)
        staff_above_below |= image[x + w * (y + staff_dist)];
    is_staff &= staff_above_below;
    output_image[x + w * y] = byte & ~ is_staff;
}

__kernel void barline_filter(__global const uchar *image,
                             int staff_thick,
                             __global uchar *output_image) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

    uchar byte = image[x + w * y];
    uchar is_barline = image[x + w * y];
    // Expect empty space past one staff thickness
    // (above or below on transpose image)
    if (0 <= y - staff_thick && y + staff_thick < h) {
        is_barline &= ~ image[x + w * (y - staff_thick)];
        is_barline &= ~ image[x + w * (y + staff_thick)];
    }
    output_image[x + w * y] = is_barline;
}
""").build()
prg.staff_removal_filter.set_scalar_arg_dtypes([
    None, np.uint32, np.uint32, None,
])
prg.barline_filter.set_scalar_arg_dtypes([
    None, np.int32, None,
])

def remove_staff(page):
    output = cla.zeros_like(page.img)
    prg.staff_removal_filter(q, page.img.shape[::-1], (16, 16),
                                page.img.data,
                                np.uint32(page.staff_thick+1),
                                np.uint32(page.staff_dist),
                                output.data).wait()
    return output

def barline_filter(page):
    no_staff = remove_staff(page)
    no_staff_T = bitimage.transpose(no_staff)
    output = cla.zeros_like(no_staff_T)
    prg.barline_filter(q, (page.img.shape[0] // 8, page.img.shape[1] * 8), # W,H
                          (16, 16),
                          no_staff_T.data,
                          np.int32(page.staff_thick+1),
                          output.data).wait()
    return bitimage.transpose(output)
