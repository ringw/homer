struct path_point {
    float cost;
    int prev;
};

kernel void staff_paths(global const uchar *image,
                        int image_w, int image_h,
                        int staff_thick,
                        float scale,
                        global struct path_point *paths,
                        int paths_w, int paths_h) {
    int worker_id = get_local_id(0);
    int num_workers = get_local_size(0);

    for (int x = 0; x < paths_w; x++) {
        if (x == 0) {
            for (int y = worker_id; y < paths_h; y += num_workers) {
                paths[x + paths_w * y].cost = 0.f;
                paths[x + paths_w * y].prev = -1;
            }
            continue;
        }
        for (int y = worker_id; y < paths_h; y += num_workers) {
            int image_x0 = convert_int_rtn(x * scale);
            int image_y0 = convert_int_rtn(y * scale);
            int image_x1 = convert_int_rtn((x + 1) * scale);
            int image_y1 = convert_int_rtn((y + 1) * scale);
            int looks_like_staff = 0;
            int any_dark = 0;
            for (int image_y = image_y0; image_y < image_y1; image_y++)
                for (int image_x = image_x0; image_x < image_x1; image_x++) {
                    if (! (0 <= image_x && image_x < image_w*8))
                        continue;
                    if (! (0 <= image_y - staff_thick && image_y + staff_thick < image_h))
                        continue;
                    int byte_x = image_x / 8;
                    int bit_x  = image_x % 8;
                    uchar center = image[byte_x + image_w * image_y];
                    uchar above  = image[byte_x + image_w * (image_y-staff_thick)];
                    uchar below  = image[byte_x + image_w * (image_y+staff_thick)];
                    if ((center & ~(above | below)) & (0x80U >> bit_x)) {
                        looks_like_staff = 1;
                        goto LOOKS_LIKE_STAFF;
                    }
                    else if (center & (0x80U >> bit_x))
                        any_dark = 1;
                }
            float base_weight;

            LOOKS_LIKE_STAFF:
            any_dark = 0;
            base_weight = (any_dark ? 4.f : 8.f) - looks_like_staff;
            float prev_cost = INFINITY;
            int prev = 0;
            for (int prev_y = y - 1; prev_y <= y + 1; prev_y++)
                if (0 <= prev_y && prev_y < paths_h) {
                    float old_cost = paths[x-1 + paths_w * prev_y].cost
                                     + ((prev_y == y) ? 0 : 1);
                    if (old_cost < prev_cost) {
                        prev_cost = old_cost;
                        prev = prev_y;
                    }
                }
            paths[x + paths_w * y].cost = prev_cost + base_weight;
            paths[x + paths_w * y].prev = prev;
        }

        // Ensure workers all process the current column before moving on
        // to the next one
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
