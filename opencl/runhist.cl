/*
 * runhist.cl - bincount all black and white vertical runs in the image
 */

#define X (1)
#define Y (0)

__kernel void runhist(__global const uchar *image,
                      __local uchar *run_starts,
                      __local int8 *temp_runs,
                      __global volatile int *run_hists,
                      __global int8 *all_runs) {
    // Store the length of the current run in temp.
    // This is done by a prefix sum in reverse (suffix sum?) starting with
    // a 1 at every pixel and updating pixels above if they are part of the
    // same run as the run below.

    // Set run_start to 1 if current pixel is different from previous
    int x = get_global_id(X);
    int y = get_global_id(Y);
    int w = get_global_size(X);
    int local_x = get_local_id(X);
    int local_y = get_local_id(Y);
    int local_w = get_local_size(X);
    uchar image_byte = image[x + y * w];
    uchar local_run_start;
    if (local_y == 0)
        local_run_start = 0xFF;
    else {
        uchar above = image[x + (y-1) * w];
        local_run_start = above ^ image_byte;
    }

    run_starts[local_x + local_y * local_w] = local_run_start;

    // Initialize every pixel to 1 (every pixel is its own run).
    temp_runs[local_x + local_y * local_w] = (int8)(1);

    mem_fence(CLK_LOCAL_MEM_FENCE);

    // When comparing two pixels, add the run count below to the one above
    // if the run count above extends to the pixel above the one below
    // and the pixel values are the same (run_start is 0 for the pixel below).
    // The updates by the suffix sum have the following shape:
    //
    // a b c d e f g h
    //  /   /   /   /
    // a b c d e f g h
    //    /       /
    //   -       -
    //  /       /
    // a b c d e f g h
    //        /
    //   -----
    //  /
    // a b c d e f g h
    //        /
    //       -
    //      /
    // a b c d e f g h
    //    /   /   /
    // a b c d e f g h

    // First stage of suffix sum: updating even elements.
    uint k = 0;
    // Compare every 2^k pixels pairwise.
    while ((1 << (k+1)) < get_local_size(Y)) {
        uint y0 = local_y;
        if (y0 % (1 << (k+1)) == 0) {
            uint y1 = y0 + (1 << k);
            // We need to compare an entire byte of the input image
            int8 run0 = temp_runs[local_x + y0 * local_w];
            int8 run1 = temp_runs[local_x + y1 * local_w];
            uchar run_start = run_starts[local_x + y1 * local_w];
            for (int b = 0; b < 8; b++) {
                if (run0[b] == y1 - y0 && (run_start & (1 << (7-b))) == 0)
                    run0[b] += run1[b];
            }
            temp_runs[local_x + y0 * local_w] = run0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        k++;
    }

    // Now atomically update the histogram for each run seen.
    int8 runlengths = temp_runs[local_x + local_y * local_w];
    int8 is_dark = (int8)image_byte;
    //is_dark[0] &= 1 << 7;
    is_dark &= (int8)(1) << (int8)(7, 6, 5, 4, 3, 2, 1, 0);
    is_dark >>= (int8)(7, 6, 5, 4, 3, 2, 1, 0);
    for (int b = 0; b < 8; b++) {
        int is_run_start = local_run_start & (1 << (7-b));
        int rl = runlengths[b];
        if (is_run_start && rl < 256 && local_y % 2 == 0) {
            atomic_inc(&run_hists[rl + is_dark[b]]);
        }
    }

    all_runs[x + y * w] =
    temp_runs[local_x + local_y * local_w];
}
