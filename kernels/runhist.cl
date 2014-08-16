/*
 * runhist.cl - bincount all black and white vertical runs in the image
 */

#define X (1)
#define Y (0)
#define NUM_COUNTS (64)

KERNEL void runhist(GLOBAL_MEM const UCHAR *image,
                    GLOBAL_MEM ATOMIC unsigned int *light_counts,
                    GLOBAL_MEM ATOMIC unsigned int *dark_counts) {
    // If our pixel is the first in its run, iterate downwards to find the
    // length and atomically update the relevant count
    int x = get_global_id(X);
    int y = get_global_id(Y);
    int w = get_global_size(X);
    int h = get_global_size(Y);

    UCHAR byte = image[x + w * y];
    UCHAR is_run = 0xFF;
    if (y > 0) {
        is_run = image[x + w * (y-1)] ^ byte;
    }

    int run_lengths[8];
    for (int i = 0; i < 8; i++) run_lengths[i] = 0;
    int some_run_inc;
    int cur_y = y + 1;
    do {
        if (! (cur_y < h)) break;
        some_run_inc = 0;
        UCHAR next_byte = image[x + w * cur_y];
        is_run &= ~(next_byte ^ byte);
        int i;
        UCHAR mask;
        for (i = 0, mask = 0x80U; mask != 0; i++, mask >>= 1) {
            if (is_run & mask) {
                run_lengths[i]++;
                some_run_inc = 1;
            }
        }
        cur_y++;
    } while (some_run_inc);

    int i;
    UCHAR mask;
    for (i = 0, mask = 0x80U; mask != 0; i++, mask >>= 1)
        if (0 < run_lengths[i] && run_lengths[i] < NUM_COUNTS) {
            if (byte & mask)
                atomic_inc(& dark_counts[run_lengths[i]]);
            else
                atomic_inc(&light_counts[run_lengths[i]]);
        }
}
