from .opencl import *

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
""").build()
prg.staff_removal_filter.set_scalar_arg_dtypes([
    None, np.uint32, np.uint32, None,
])

def remove_staff(page):
    output = cla.zeros_like(page.img)
    prg.staff_removal_filter(q, page.img.shape[::-1], (16, 16),
                                page.img.data,
                                np.uint32(page.staff_thick+1),
                                np.uint32(page.staff_dist),
                                output.data).wait()
    return output
