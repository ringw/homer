/*
 * staffsize.cl - bincount all black and white vertical runs in the image
 */

#define X (1)
#define Y (0)
#define NUM_COUNTS (64)

KERNEL void dark_hist(GLOBAL_MEM const UCHAR *image,
                      GLOBAL_MEM ATOMIC unsigned int *dark_counts) {
    // If our pixel is the first in its run, iterate downwards to find the
    // length and atomically update the relevant count
    int x = get_global_id(X);
    int y = get_global_id(Y);
    int w = get_global_size(X);
    int h = get_global_size(Y);

    UCHAR byte = image[x + w * y];
    int8 bits = fill_int8(byte);
    bits >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
    bits &= fill_int8(0x1);
    // Track run only if it is black
    int8 is_run = bits;
    if (y > 0) {
        int8 above_bits = fill_int8(image[x + w * (y-1)]);
        above_bits >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
        above_bits &= fill_int8(0x1);
        is_run &= (above_bits ^ bits);
    }

    int8 run_lengths = is_run;
    float4 ones = make_float4(1.f,1.f,1.f,1.f);
    int cur_y = y + 1;
    while (dot(ones, convert_float4(is_run.s0123))
            + dot(ones, convert_float4(is_run.s4567)) > 0
           && cur_y < h) {
        int8 next_bits = fill_int8(image[x + w * cur_y]);
        next_bits >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
        next_bits &= fill_int8(0x1);
        is_run &= ~(next_bits ^ bits);
        run_lengths += is_run;
        cur_y += 1;
    }

    union {
        int8 v;
        int a[8];
    } rl_u, bit_u;
    rl_u.v = run_lengths;
    bit_u.v = bits;
    
    for (int i = 0; i < 8; i++)
        if (0 < rl_u.a[i] && rl_u.a[i] < NUM_COUNTS) {
            atomic_inc(&dark_counts[rl_u.a[i]]);
        }
}

KERNEL void light_hist(GLOBAL_MEM const UCHAR *image,
                       int staff_thick, // argmax(dark_counts)
                       GLOBAL_MEM ATOMIC unsigned int *light_counts) {
    // If our pixel is the first in its run, iterate downwards to find the
    // length and atomically update the relevant count
    // Also, must have light space 2*staff_thick above and below run
    int x = get_global_id(X);
    int y = get_global_id(Y);
    int w = get_global_size(X);
    int h = get_global_size(Y);

    UCHAR byte = image[x + w * y];
    int8 bits = fill_int8(byte);
    bits >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
    bits &= fill_int8(0x1);
    // Track run only if it is white
    int8 is_run = fill_int8(~byte);
    is_run >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
    is_run &= fill_int8(0x1);
    if (y > 0) {
        int8 above_bits = fill_int8(image[x + w * (y-1)]);
        above_bits >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
        above_bits &= fill_int8(0x1);
        is_run &= (above_bits ^ bits);
    }
    if (y > 2*staff_thick) {
        // Check for white space above run
        int8 above_bits = fill_int8(~image[x + w * (y - 2*staff_thick)]);
        above_bits >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
        above_bits &= fill_int8(0x1);
        is_run &= (above_bits ^ bits);
    }
    else
        is_run = 0; // assume not tracking runs on very top or bottom

    int8 run_lengths = is_run;
    float4 ones = make_float4(1.f,1.f,1.f,1.f);
    int cur_y = y + 1;
    while (dot(ones, convert_float4(is_run.s0123))
            + dot(ones, convert_float4(is_run.s4567)) > 0
           && cur_y < h) {
        int8 next_bits = fill_int8(image[x + w * cur_y]);
        next_bits >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
        next_bits &= fill_int8(0x1);
        is_run &= ~(next_bits ^ bits);
        run_lengths += is_run;
        cur_y += 1;
    }

    union {
        int8 v;
        int a[8];
    } rl_u, bit_u;
    rl_u.v = run_lengths;
    bit_u.v = bits;
    
    for (int i = 0; i < 8; i++)
        if (0 < rl_u.a[i] && rl_u.a[i] < NUM_COUNTS) {
            // Check 2*staff_thick under run
            int y_below = y + rl_u.a[i] + 2*staff_thick;
            int light_below = 1;
            if (y_below < h) {
                UCHAR byte_below = image[x + w * y_below];
                if (byte_below & (0x80U >> i))
                    light_below = 0;
            }
            else // too close to bottom edge to track
                light_below = 0;
            if (light_below)
                atomic_inc(&light_counts[rl_u.a[i]]);
        }
}

KERNEL void cardoso_rebelo_staffdist(GLOBAL_MEM const UCHAR *image,
                  LOCAL_MEM UCHAR *image_cache,
                  GLOBAL_MEM ATOMIC unsigned int *staff_dist_hist) {
    // If our pixel is the first in its run, iterate downwards for 2 runs.
    int x = get_global_id(X);
    int y = get_global_id(Y);
    int w = get_global_size(X);
    int h = get_global_size(Y);

    int x_local = get_local_id(X);
    int y_local = get_local_id(Y);
    int w_local = get_local_size(X);
    image_cache[x_local + w_local * y_local] = image[x + w * y];

    for (uchar mask = 0x80U; mask; mask >>= 1) {
        int y_ = y;
        int num_runs = 0;
        if (y != 0) {
            if (y_local > 0) {
                if ((image_cache[x_local + w_local * (y_local-1)]
                    ^ image_cache[x_local + w_local * y_local]) & mask == 0) continue;
            }
            else
                if ((image[x + w * (y-1)]
                     ^ image_cache[x_local + w_local * y_local]) & mask == 0) continue;
        }
        do {
            y_++;
            if ((image[x + w * (y_-1)] ^ image[x + w * y_]) & mask)
                num_runs++;
        } while (y_ < h && num_runs < 2);
        if (y_ < h && (y_ - y) < NUM_COUNTS)
            atomic_inc(&staff_dist_hist[y_ - y]);
    }
}
