KERNEL void staff_removal(GLOBAL_MEM const int2 *staves,
                          int staff_thick, int staff_dist,
                          GLOBAL_MEM UCHAR *img,
                          int w, int h,
                          GLOBAL_MEM int2 *refined_staves,
                          int refined_num_points) {
    int num_points = get_global_size(0);
    int num_staves = get_global_size(1);
    int segment_num = get_global_id(0);
    int staff_num = get_global_id(1);

    int remove_staff = 1;
    if (refined_num_points < 0) {
        remove_staff = 0;
        refined_num_points = -refined_num_points;
    }

    if (segment_num == 0) {
        // Mask refined_staves
        for (int i = 0; i < refined_num_points; i++) {
            refined_staves[i + refined_num_points*staff_num] = make_int2(-1,-1);
        }
    }
    if (segment_num + 1 == num_points)
        return;
    int2 p0 = staves[segment_num     + num_points * staff_num];
    int2 p1 = staves[segment_num + 1 + num_points * staff_num];
    if (p0.x < 0 || p1.x < 0)
        return;

    // Fudge x-values to nearest byte
    for (int byte_x = p0.x / 8; byte_x <= p1.x / 8 && byte_x < w; byte_x++) {
        int y = p0.y + (p1.y - p0.y) * (byte_x*8 - p0.x) / (p1.x - p0.x);

        // Try to refine y-value by searching an small area
        UCHAR buf[64];
        int dy = MIN(31, (staff_thick+1)/2);
        int y0 = MAX(0, y - dy);
        int y1 = MIN(h, y + dy + 1);
        for (int y_ = y0, i = 0; y_ < y1; y_++, i++)
            buf[i] = img[byte_x + w * y_];

        // At each x position in the byte, search for a short run
        int run_center_y[8];
        int num_runs = 0;
        for (int bit = 0; bit < 8; bit++)
            run_center_y[bit] = -1;

        for (int bit = 0; bit < 8; bit++) {
            UCHAR mask = 0x80U >> bit;
            int best_run_y = -1;

            int cur_run = 0;
            for (int y_ = y0, i = 0; y_ < y1; y_++, i++) {
                if (buf[i] & mask)
                    cur_run++;
                else if (cur_run) {
                    if (cur_run < staff_thick*2) {
                        int y_center = y_ + (-cur_run / 2);
                        if (best_run_y == -1
                            || ABS(best_run_y - y) > ABS(y_center - y))
                            best_run_y = y_center;
                    }
                    cur_run = 0;
                }
            }
            if (best_run_y >= 0)
                run_center_y[num_runs++] = best_run_y;
        }

        if (num_runs == 0)
            continue;
        // A really inefficient median finding algorithm
        // Set the minimum element to -1 for enough iterations
        int median_ind;
        for (int count = 0; count <= num_runs/2; count++) {
            median_ind = -1;
            // Remove the last minimum
            if (count)
                run_center_y[median_ind] = -1;
            for (int elem = 0; elem < num_runs; elem++) {
                int value = run_center_y[elem];
                if (value >= 0 && (median_ind == -1 || value < median_ind))
                    median_ind = elem;
            }
        }

        int y_refined = run_center_y[median_ind];

        int lines[5] = {y_refined - staff_dist*2,
                        y_refined - staff_dist,
                        y_refined,
                        y_refined + staff_dist,
                        y_refined + staff_dist*2};
        if (! (0 <= lines[0] - staff_thick && lines[4] + staff_thick < h))
            continue;

        UCHAR is_staff = 0xFF;
        UCHAR found_line[5];
        for (int i = 0; i < 5; i++) {
            found_line[i] = 0;
            for (int dy = -staff_thick; dy <= staff_thick; dy++)
                found_line[i] |= img[byte_x + w * (lines[i] + dy)];
            is_staff &= found_line[i];
        }

        UCHAR mask[5];
        for (int i = 0; i < 5; i++) {
            mask[i] = ~ is_staff;
            // Must have empty space +- staff_thick
            mask[i] |= img[byte_x + w * (lines[i] - staff_thick)];
            mask[i] |= img[byte_x + w * (lines[i] + staff_thick)];
        }
        UCHAR some_space = 0;
        for (int i = 0; i < 5; i++)
            some_space |= found_line[i] & mask[i];
        is_staff &= some_space;

        if (byte_x < refined_num_points && is_staff != 0)
            refined_staves[byte_x + refined_num_points * staff_num] =
                make_int2(byte_x * 8, y_refined);

        if (! remove_staff)
            continue;
        for (int i = 0; i < 5; i++) {
            for (int dy = -staff_thick/2; dy <= staff_thick/2; dy++)
                img[byte_x + w * (lines[i] + dy)] &= mask[i];
        }
    }
}
