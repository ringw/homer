KERNEL void staff_removal(GLOBAL_MEM const int2 *staves,
                          int staff_thick, int staff_dist,
                          GLOBAL_MEM UCHAR *img,
                          int w, int h) {
    int num_points = get_global_size(0);
    int num_staves = get_global_size(1);
    int segment_num = get_global_id(0);
    int staff_num = get_global_id(1);

    if (segment_num + 1 == num_points)
        return;
    int2 p0 = staves[segment_num     + num_points * staff_num];
    int2 p1 = staves[segment_num + 1 + num_points * staff_num];
    if (p0.x < 0 || p1.x < 0)
        return;

    // Fudge x-values to nearest byte
    for (int byte_x = p0.x / 8; byte_x <= p1.x / 8 && byte_x < w; byte_x++) {
        int y = p0.y + (p1.y - p0.y) * (byte_x*8 - p0.x) / (p1.x - p0.x);
        int lines[5] = {y - staff_dist*2,
                        y - staff_dist,
                        y,
                        y + staff_dist,
                        y + staff_dist*2};
        if (! (0 <= lines[0] - staff_thick && lines[4] + staff_thick < h))
            continue;
        UCHAR is_staff = 0xFF;
        for (int i = 0; i < 5; i++) {
            UCHAR found_line = 0;
            for (int dy = -staff_thick/2; dy <= staff_thick/2; dy++)
                found_line |= img[byte_x + w * (lines[i] + dy)];
            is_staff &= found_line;
        }

        for (int i = 0; i < 5; i++) {
            UCHAR mask = ~ is_staff;
            // Must have empty space +- staff_thick
            mask |= img[byte_x + w * (lines[i] - staff_thick)];
            mask |= img[byte_x + w * (lines[i] + staff_thick)];

            for (int dy = -staff_thick/2; dy <= staff_thick/2; dy++)
                img[byte_x + w * (lines[i] + dy)] &= mask;
        }
    }
}
