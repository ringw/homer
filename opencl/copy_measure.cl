__kernel void copy_measure(__global const uchar *image,
                           int2 image_size,
                           __global const uint *top_ys,
                           int x_space_top,
                           __global const uint *bottom_ys,
                           int x_space_bottom,
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
        int boundary_segment_top = image_x / x_space_top;
        int bound_top_left = top_ys[boundary_segment_top];
        int bound_top_right = top_ys[boundary_segment_top + 1];
        int bound_top = bound_top_left + (bound_top_right - bound_top_left)
                                            * (x % x_space_top) / x_space_top;
        int boundary_segment_bottom = image_x / x_space_bottom;
        int bound_bottom_left = bottom_ys[boundary_segment_bottom];
        int bound_bottom_right = bottom_ys[boundary_segment_bottom + 1];
        int bound_bottom = bound_bottom_left
                                + (bound_bottom_right - bound_bottom_left)
                                            * (x % x_space_bottom)
                                              / x_space_bottom;
        if (bound_top <= image_y && image_y < bound_bottom) {
            if (image_byte & mask)
                output |= mask;
        }
        mask <<= 1;
    }
    measure[byte_x + measure_bounds.s2 * y] = output;
}
