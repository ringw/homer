from .opencl import *
import numpy as np

uint4 = cl.tools.get_or_register_dtype("uint4")
prg = cl.Program(cx, """
__kernel void copy_measure(__global const uchar *image,
                           uint2 image_size,
                           __global const uint *top_ys,
                           __global const uint *bottom_ys,
                           uint x_space,
                           __global uchar *measure,
                           uint4 measure_bounds) {
    // top_boundary and bottom_boundary must have the same x-values
    // which are evenly spaced by x_space.
    // Each worker outputs one byte of measure.
    uint x_byte = get_global_id(0);
    uint y = get_global_id(1);
    uint image_x = measure_bounds.s0 / 8 + x_byte;
    uint image_y = measure_bounds.s1 + y;
    uchar image_byte = 0;
    if (0 <= image_x && image_x < image_size.x
        && 0 <= image_y && image_y < image_size.y) {
        image_byte = image[image_x + image_size.x * image_y];
    }

    // Mask out portion of image_byte for each boundary segment
    for (uint boundary_p0 = x_byte * 8 / x_space;
         boundary_p0 * x_space < (image_x + 1) * 8;
         boundary_p0++) {
        uint boundary_p1 = boundary_p0 + 1;
        // Arithmetic shift from left
        uchar boundary_mask = (char)0x80
                    >> MIN(7, boundary_p1 * x_space - x_byte * 8);
        boundary_mask &= (uchar)0xFF
                    >> MAX(0, (int)boundary_p0 * x_space - x_byte * 8);
        image_byte &= mask;
    }

    measure[x_byte + measure_bounds.s2 * y] = image_byte;
}
""")
def get_measure(page, staff, measure):
    for bar_start, bar_stop, barlines in page.barlines:
        if bar_start <= staff and staff < bar_stop:
            break
    else:
        raise Exception("Staff not found in barlines")
    if measure + 1 >= len(barlines):
        raise Exception("Measure out of bounds")
    # Round down measure start x
    x0 = barlines[measure] & -8
    # Round up measure end x
    x1 = -(-barlines[measure] & -8)

    # Round up staff start y
    y0 = page.boundaries[staff][:, 1].min() & -8
    # Round down staff end y
    y1 = -(-page.boundaries[staff+1][:, 1].max() & -8)
