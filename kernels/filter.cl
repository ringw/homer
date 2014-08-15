#define X (0)
#define Y (1)

KERNEL void staff_center_filter(GLOBAL_MEM const UCHAR *image,
                                int staff_dist,
                                GLOBAL_MEM UCHAR *staff) {
    // Ensure a given pixel has dark pixels above and below where we would
    // expect if it were the center of a staff, then update the center pixel.
    int x = get_global_id(X);
    int y = get_global_id(Y);
    int w = get_global_size(X);
    int h = get_global_size(Y);
    
    UCHAR staff_byte = image[x + y * w];

    for (int i = -2; i <= 2; i++) {
        if (i == 0)
            continue;
        UCHAR found_point = 0x0;
        // Search within 3 points of expected distance
        for (int d = -3; d <= 3; d++) {
            int point_y = y + i*staff_dist + d;
            if (0 <= point_y && point_y < h)
                found_point |= image[x + point_y * w];
        }
        staff_byte &= found_point;
    }

    staff[x + y * w] = staff_byte;
}

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
