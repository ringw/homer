__kernel void staff_removal_filter(__global const uchar *image,
                                   int staff_thick,
                                   int staff_dist,
                                   __global uchar *output_image) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

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
