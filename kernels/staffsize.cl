/*
 * staffsize.cl - staff size estimation
 * An implementation of the algorithm:
 * Cardoso JS, Rebelo A (2010) Robust staffline thickness and distance
 * estimation in binary and gray-level music scores.
 * In: Proceedings of the twentieth international conference on pattern
 * recognition, pp 1856-1859
 */

#define X (1)
#define Y (0)
#define NUM_COUNTS (64)

// Histogram of staff_dist values for 2 consecutive runs
KERNEL void staff_dist_hist(GLOBAL_MEM const UCHAR *image,
                            GLOBAL_MEM ATOMIC unsigned int *dist_counts) {
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
    // Decrement is_run when we switch pixel value, so we end up histogramming
    // the sums of all consecutive pairs of runs
    int8 is_run;
    if (y > 0) {
        int8 above_bits = fill_int8(image[x + w * (y-1)]);
        above_bits >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
        above_bits &= fill_int8(0x1);
        is_run = (above_bits ^ bits) << 1;
    }
    else
        is_run = fill_int8(0x0);

    int8 run_lengths = is_run;
    int cur_y = y + 1;
    while ((is_run.s0 || is_run.s1 || is_run.s2 || is_run.s3
            || is_run.s4 || is_run.s5 || is_run.s6 || is_run.s7)
           && cur_y < h && cur_y - y < NUM_COUNTS) {
        int8 next_bits = fill_int8(image[x + w * cur_y]);
        next_bits >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
        next_bits &= fill_int8(0x1);
        is_run >>= 0x1 & (next_bits ^ bits);
        // Increment run_lengths if is_run is 2 or 1
        run_lengths += (is_run ^ (is_run >> 1)) & 0x1;
        cur_y += 1;
        bits = next_bits;
    }

    // Need to convert vector to int array for looping
    union {
        int8 v;
        int a[8];
    } rl_u;
    rl_u.v = run_lengths;
    
    for (int i = 0; i < 8; i++)
        if (0 < rl_u.a[i] && rl_u.a[i] < NUM_COUNTS) {
            atomic_inc(&dist_counts[rl_u.a[i]]);
        }
}

// Using known staff_dist value, get staff_thick value
KERNEL void staff_thick_hist(GLOBAL_MEM const UCHAR *image,
                             int staff_dist,
                             GLOBAL_MEM ATOMIC unsigned int *thick_counts) {
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
    // Decrement is_run when we switch pixel value, so we end up histogramming
    // the sums of all consecutive pairs of runs
    int8 is_run = fill_int8(0x2);
    if (y > 0) {
        int8 above_bits = fill_int8(image[x + w * (y-1)]);
        above_bits >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
        above_bits &= fill_int8(0x1);
        is_run &= (above_bits ^ bits) << 1;
    }

    // Store both consecutive 2 runs (staff_dist) and dark run only
    int8 run_lengths = is_run;
    int8 dark_lengths = 0x1 & ~bits;
    int cur_y = y + 1;
    while ((is_run.s0 || is_run.s1 || is_run.s2 || is_run.s3
            || is_run.s4 || is_run.s5 || is_run.s6 || is_run.s7)
           && cur_y < h && cur_y - y < NUM_COUNTS) {
        int8 next_bits = fill_int8(image[x + w * cur_y]);
        next_bits >>= make_int8(7, 6, 5, 4, 3, 2, 1, 0);
        next_bits &= fill_int8(0x1);
        is_run >>= 0x1 & (next_bits ^ bits);
        // Increment run_lengths if is_run is 2 or 1
        run_lengths += (is_run ^ (is_run >> 1)) & 0x1;
        dark_lengths += (is_run ^ (is_run >> 1)) & 0x1 & next_bits;
        cur_y += 1;
        bits = next_bits;
    }

    union {
        int8 v;
        int a[8];
    } dist_u, thick_u;
    dist_u.v = run_lengths;
    thick_u.v = dark_lengths;
    
    for (int i = 0; i < 8; i++)
        if (dist_u.a[i] == staff_dist
            && 0 < thick_u.a[i] && thick_u.a[i] < NUM_COUNTS) {
            atomic_inc(&thick_counts[thick_u.a[i]]);
        }
}
