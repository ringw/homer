from .opencl import *
import numpy as np

uint2 = cl.tools.get_or_register_dtype("uint2")
uint4 = cl.tools.get_or_register_dtype("uint4")
prg = cl.Program(cx, """
__kernel void copy_measure(__global const uchar *image,
                           int2 image_size,
                           __global const uint *top_ys,
                           __global const uint *bottom_ys,
                           int x_space,
                           __global uchar *measure,
                           int4 measure_bounds) {
    // top_boundary and bottom_boundary must have the same x-values
    // which are evenly spaced by x_space.
    int byte_x = get_global_id(0);
    int y = get_global_id(1);
    int image_byte_x = measure_bounds.s0 + byte_x;
    int image_y = measure_bounds.s1 + y;

    uchar output = 0;
    uchar mask = 0x01;
    uchar image_byte = image[image_byte_x + image_size.x * image_y];
    for (int b = 7; b >= 0; b--) {
        int x = byte_x * 8 + b;
        int image_x = image_byte_x * 8 + b;
        // Determine whether this pixel is within the boundaries
        int boundary_segment = image_x / x_space;
        int bound_top_left = top_ys[boundary_segment];
        int bound_top_right = top_ys[boundary_segment + 1];
        int bound_top = bound_top_left + (bound_top_right - bound_top_left)
                                            * (x % x_space) / x_space;
        int bound_bottom_left = bottom_ys[boundary_segment];
        int bound_bottom_right = bottom_ys[boundary_segment + 1];
        int bound_bottom = bound_bottom_left
                                + (bound_bottom_right - bound_bottom_left)
                                            * (x % x_space) / x_space;
        if (bound_top <= image_y && image_y < bound_bottom) {
            if (image_byte & mask)
                output |= mask;
        }
        mask <<= 1;
    }
    measure[byte_x + measure_bounds.s2 * y] = output;
}
""").build()
prg.copy_measure.set_scalar_arg_dtypes([
    None, uint2, None, None, np.uint32, None, uint4
])
def get_measure(page, staff, measure):
    for bar_start, bar_stop, barlines in page.barlines:
        if bar_start <= staff and staff < bar_stop:
            break
    else:
        raise Exception("Staff not found in barlines")
    if measure + 1 >= len(barlines):
        raise Exception("Measure out of bounds")
    # Round down measure start x
    x0 = barlines[measure, [0,1]].min() & -8
    # Round up measure end x
    x1 = -(-barlines[measure + 1, [0,1]].max() & -8)

    # Round up staff start y
    y0 = page.boundaries[staff][:, 1].min() & -8
    # Round down staff end y
    y1 = -(-page.boundaries[staff+1][:, 1].max() & -8)

    measure_pixel_size = (y1 - y0, (x1 - x0) // 8)
    measure_size = tuple(-(-i & -16) for i in measure_pixel_size)
    measure = cla.zeros(q, measure_size, np.uint8)
    device_b0 = cla.to_device(q, page.boundaries[staff][:, 1].astype(np.uint32))
    device_b1 = cla.to_device(q, page.boundaries[staff + 1][:, 1]
                                     .astype(np.uint32))
    prg.copy_measure(q, measure.shape[::-1], (1, 1),
                        page.img.data,
                        np.array(page.img.shape[::-1],
                                 np.uint32).view(uint2)[0],
                        device_b0.data, device_b1.data,
                        np.uint32(page.boundaries[staff][1, 0]
                                    - page.boundaries[staff][0, 0]),
                        measure.data,
                        np.array([x0 // 8, y0] + list(measure_size[::-1]),
                                 np.uint32).view(uint4)[0]).wait()
    return measure
