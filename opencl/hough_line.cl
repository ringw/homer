/*
 * hough_line: Hough transform for lines
 * (image_w, imageH) are the dimensions in bytes of the bit-packed image
 * theta is angle to the horizontal
 * bins are shape (len(theta), len(rho))
 * global size should be (len(rho) * num_workers, len(theta))
 * local size may be (num_workers, 1), multiple workers will sum pixels in parallel
 */
__kernel void hough_line(__global const uchar *image,
                         uint image_w, uint image_h,
                         uint rhores,
                         __global const float *cos_thetas,
                         __global const float *sin_thetas,
                         __local float *worker_sums,
                         __global float *bins) {
    uint rho = get_group_id(0);
    uint num_rho = get_num_groups(0);
    uint theta = get_global_id(1);

    float rho_val = rho * rhores;
    float cos_theta = cos_thetas[theta];
    float sin_theta = sin_thetas[theta];

    // Multiple workers help sum up the same rho
    uint worker_id = get_local_id(0);
    uint num_workers = get_local_size(0);

    float worker_sum = 0.f;
    // Sum each x-byte position. As an approximation, assume if the left
    // corner is parameterized as (rho, theta) then we can sum up the whole byte
    for (int x = 0; x < image_w; x += num_workers) {
        float x_left_val = x * 8;
        float y_val = (rho_val - x_left_val * sin_theta) / cos_theta;
        int y = convert_int_rtn(y_val);

        if (0 <= x && x < image_w && 0 <= y && y < image_h) {
            uchar byte = image[x + image_w * y];
            uint8 bits = (uint8)byte;
            bits >>= (uint8)(7, 6, 5, 4, 3, 2, 1, 0);
            bits &= (uint8)(0x1);
            // Sum using float dot product (faster)
            float8 fbits = convert_float8(bits);
            float4 one = (float4)(1.f);
            worker_sum += dot(fbits.s0123, one);
            worker_sum += dot(fbits.s4567, one);
        }
    }

    if (num_workers > 1) {
        worker_sums[worker_id] = worker_sum;
        mem_fence(CLK_LOCAL_MEM_FENCE);
        if (worker_id == 0) {
            // Sum all partial sums into global bin
            float global_sum = 0.f;
            for (int i = 0; i < num_workers; i++)
                global_sum += worker_sums[i];
            bins[rho + num_rho * theta] = global_sum;
        }
    }
    else {
        // We are the only worker
        bins[rho + num_rho * theta] = worker_sum;
    }
}
