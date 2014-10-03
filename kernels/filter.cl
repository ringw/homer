#define X (0)
#define Y (1)

KERNEL void staff_removal_filter(GLOBAL_MEM const UCHAR *image,
                                 int staff_thick,
                                 int staff_dist,
                                 GLOBAL_MEM UCHAR *output_image) {
    int x = get_global_id(X);
    int y = get_global_id(Y);
    int w = get_global_size(X);
    int h = get_global_size(Y);

    UCHAR byte = image[x + w * y];
    UCHAR is_staff = byte;
    // Expect empty space above or below
    if (0 <= y - staff_thick && y + staff_thick < h) {
        is_staff &= ~ image[x + w * (y - staff_thick)];
        is_staff &= ~ image[x + w * (y + staff_thick)];
    }
    // Expect another staff either above or below
    UCHAR staff_above_below = 0;
    if (0 <= y - staff_dist)
        staff_above_below |= image[x + w * (y - staff_dist)];
    if (y + staff_dist < h)
        staff_above_below |= image[x + w * (y + staff_dist)];
    is_staff &= staff_above_below;
    output_image[x + w * y] = byte & ~ is_staff;
}

KERNEL void barline_filter(GLOBAL_MEM const UCHAR *image,
                           int staff_thick,
                           GLOBAL_MEM UCHAR *output_image) {
    int x = get_global_id(X);
    int y = get_global_id(Y);
    int w = get_global_size(X);
    int h = get_global_size(Y);

    UCHAR byte = image[x + w * y];
    UCHAR is_barline = image[x + w * y];
    // Expect empty space past one staff thickness
    // (above or below on transpose image)
    if (0 <= y - staff_thick && y + staff_thick < h) {
        is_barline &= ~ image[x + w * (y - staff_thick)];
        is_barline &= ~ image[x + w * (y + staff_thick)];
    }
    output_image[x + w * y] = is_barline;
}
