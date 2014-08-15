/*
 * runhist.cl - bincount all black and white vertical runs in the image
 */

#define X (1)
#define Y (0)
#define NUM_COUNTS (64)

KERNEL void runhist(GLOBAL_MEM const UCHAR *image,
                      GLOBAL_MEM volatile int *light_counts,
                      GLOBAL_MEM volatile int *dark_counts) {
    // If our pixel is the first in its run, iterate downwards to find the
    // length and atomically update the relevant count
    int x = get_global_id(X);
    int y = get_global_id(Y);
    int w = get_global_size(X);
    int h = get_global_size(Y);

    UCHAR byte = image[x + w * y];
    int8 bits = (int8)byte;
    bits >>= (int8)(7, 6, 5, 4, 3, 2, 1, 0);
    bits &= (int8)0x1;
    int8 is_run = (int8)(1);
    if (y > 0) {
        int8 above_bits = (int8)image[x + w * (y-1)];
        above_bits >>= (int8)(7, 6, 5, 4, 3, 2, 1, 0);
        above_bits &= (int8)0x1;
        is_run &= (above_bits ^ bits);
    }

    int8 run_lengths = is_run;
    float4 ones = (float4)(1.f);
    int cur_y = y + 1;
    while (dot(ones, convert_float4(is_run.s0123))
            + dot(ones, convert_float4(is_run.s4567)) > 0
           && cur_y < h) {
        int8 next_bits = (int8)image[x + w * cur_y];
        next_bits >>= (int8)(7, 6, 5, 4, 3, 2, 1, 0);
        next_bits &= (int8)0x1;
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
            if (bit_u.a[i])
                atomic_inc(& dark_counts[rl_u.a[i]]);
            else
                atomic_inc(&light_counts[rl_u.a[i]]);
        }
}
