#define X (0)
#define Y (1)

__kernel void staffpoints(__global const uchar *image,
                          int staff_dist,
                          __global uchar *staff) {
    // Ensure a given pixel has dark pixels above and below where we would
    // expect if it were the center of a staff, then update the center pixel.
    int x = get_global_id(X);
    int y = get_global_id(Y);
    int w = get_global_size(X);
    int h = get_global_size(Y);
    
    uchar staff_byte = image[x + y * w];

    for (int i = -2; i <= 2; i++) {
        if (i == 0)
            continue;
        uchar found_point = 0x0;
        // Search within 2 points of expected distance
        for (int d = -3; d <= 3; d++) {
            int point_y = y + i*staff_dist + d;
            if (0 <= point_y && point_y < h)
                found_point |= image[x + point_y * w];
        }
        staff_byte &= found_point;
    }

    staff[x + y * w] = staff_byte;
}
