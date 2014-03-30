/*
 * runhist.cl - bincount all black and white vertical runs in the image
 */

#define X (1)
#define Y (0)
#define NUM_COUNTS (64)

__kernel void runhist(__global const uchar *image,
                      __global volatile uint *light_counts,
                      __global volatile uint *dark_counts) {
    // If our pixel is the first in its run, iterate downwards to find the
    // length and atomically update the relevant count
    uint x = get_global_id(X);
    uint y = get_global_id(Y);
    uint w = get_global_size(X);
    uint h = get_global_size(Y);

    uchar byte = image[x + w * y];
    uint8 bits = (uint8)byte;
    bits >>= (uint8)(7, 6, 5, 4, 3, 2, 1, 0);
    bits &= (uint8)0x1;
    uint8 is_run = (uint8)(1);
    if (y > 0) {
        uint8 above_bits = (uint8)image[x + w * (y-1)];
        above_bits >>= (uint8)(7, 6, 5, 4, 3, 2, 1, 0);
        above_bits &= (uint8)0x1;
        is_run &= (above_bits ^ bits);
    }

    uint8 run_lengths = is_run;
    float4 ones = (float4)(1.f);
    uint cur_y = y + 1;
    while (dot(ones, convert_float4(is_run.s0123))
            + dot(ones, convert_float4(is_run.s4567)) > 0
           && cur_y < h) {
        uint8 next_bits = (uint8)image[x + w * cur_y];
        next_bits >>= (uint8)(7, 6, 5, 4, 3, 2, 1, 0);
        next_bits &= (uint8)0x1;
        is_run &= ~(next_bits ^ bits);
        run_lengths += is_run;
        cur_y += 1;
    }

    union {
        uint8 v;
        uint a[8];
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
