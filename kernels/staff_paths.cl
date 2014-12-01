struct path_point {
    float cost;
    int prev;
};

// image should be a scaled down grayscale version
KERNEL void staff_paths(GLOBAL_MEM const UCHAR *image,
                        int image_w, int image_h,
                        GLOBAL_MEM struct path_point *paths) {
    int worker_id = get_local_id(0);
    int num_workers = get_local_size(0);

    for (int x = 0; x < image_w; x++) {
        if (x == 0) {
            for (int y = worker_id; y < image_h; y += num_workers) {
                paths[x + image_w * y].cost = 0.f;
                paths[x + image_w * y].prev = -1;
            }
        }
        else {
            for (int y = worker_id; y < image_h; y += num_workers) {
                UCHAR our_value = image[x + image_h * y];
                float base_weight = 4.f - (float)our_value / 255.f;
                // Calculate weight from pixel to left
                int prev = y;
                UCHAR left_value = image[x-1 + image_w * y];
                float weight = paths[x-1 + image_w * prev].cost
                               + base_weight - (float)left_value / 255.f;
                for (int i = 0; i < 2; i++) {
                    int prev_y = i == 0 ? y-1 : y+1;
                    if (0 <= prev_y && prev_y < image_h) {
                        left_value = image[x-1 + image_h * prev_y];
                        float new_weight =
                            paths[x-1 + image_h * prev_y].cost +
                            (base_weight - (float)left_value / 255.f) * 1.414f;
                        if (new_weight < weight) {
                            prev = prev_y;
                            weight = new_weight;
                        }
                    }
                }
                paths[x + image_w * y].cost = weight;
                paths[x + image_w * y].prev = prev;
            }
        }

        // Ensure workers all process the current column before moving on
        // to the next one
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

KERNEL void find_stable_paths(GLOBAL_MEM const struct path_point *paths,
                              int w,
                              GLOBAL_MEM ATOMIC int *stable_path_end) {
    int y1 = get_global_id(0);
    // Initialize stable_path_end
    stable_path_end[y1] = -1;
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Trace back shortest path from (x1, y1)
    int y = y1;
    for (int x = w-1; x > 0; x--)
        y = paths[x + w * y].prev;

    // Update stable_path_end[y] with y1 as long as the path is shorter
    // than the one currently stored there
    float our_cost = paths[w-1 + w * y1].cost;
    int cur_y1;
    do {
        cur_y1 = stable_path_end[y];
        if (cur_y1 >= 0 && paths[w-1 + w * cur_y1].cost >= our_cost)
            break;
    } while (atomic_cmpxchg(&stable_path_end[y], cur_y1, y1) != cur_y1);
}

// stable_path_end must be packed by removing invalid y < 0
KERNEL void extract_stable_paths(GLOBAL_MEM const struct path_point *paths,
                                 int w,
                                 GLOBAL_MEM const int *stable_path_end,
                                 GLOBAL_MEM int *stable_paths) {
    int path_ind = get_global_id(0);
    int y = stable_path_end[path_ind];
    for (int x = w-1; x >= 0; x--) {
        stable_paths[x + w * path_ind] = y;
        y = paths[x + w * y].prev;
    }
}

KERNEL void remove_paths(GLOBAL_MEM UCHAR *image,
                         int w, int h,
                         GLOBAL_MEM const int *stable_paths,
                         int path_num_points,
                         int path_scale,
                         GLOBAL_MEM int *pixel_sums) {
    if (path_scale != 2) return; // XXX

    int path = get_global_id(0);
    int our_sum = 0;
    for (int x = 0; x*path_scale < w*8; x++) {
        UCHAR byte_mask = 0xC0U >> ((x*path_scale) % 8);
        int y_center = stable_paths[x + path_num_points * path]*path_scale;
        int any_dark = 0;
        for (int y = y_center - 5; y < y_center + 5 + path_scale; y++) {
            if (0 <= y && y < h) {
                if (image[(x*path_scale)/8 + w * y] & byte_mask)
                    any_dark = 1;
                image[(x*path_scale)/8 + w * y] &= ~ byte_mask;
            }
        }

        our_sum += any_dark;
    }

    pixel_sums[path] = our_sum;
}
