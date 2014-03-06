__kernel void hough_line(__global const uchar *input,
                       __global const float *tan_theta,
                       int rhores, int nbins,
                       __local int8 *temp,
                       __global volatile int *bins) {
    // Get top left x and y of block
    int blockW = get_local_size(0)*8;
    int blockH = get_local_size(1);
    int x0 = blockW * get_group_id(0);
    int y0 = blockH * get_group_id(1);
    int theta_num = get_global_id(2);
    float tt = tan_theta[theta_num];

    // Calculate rho for (x0, y0)
    // Maximum rho in block is less than minrho + (blockW + blockH)/rhores
    int minrho = convert_int_rtn((-tt * x0 + y0) / rhores);
    int numrho = convert_int_rtn((blockW + blockH) / rhores);

    int num_workers = get_local_size(0) * get_local_size(1);
    int worker_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int input_ind = get_global_id(1) * get_global_size(0) + get_global_id(0);

    int8 blockX = {0, 1, 2, 3, 4, 5, 6, 7};
    blockX += convert_int8(get_local_id(0) * 8);
    int blockY = get_local_id(1);
    float8 rhovals = (-tt * convert_float8(blockX) + blockY) / rhores;
    int8 rhoind = convert_int8(rhovals);

    // Mask rhoind where image is zero to a negative value so it's not counted
    uchar8 val = convert_uchar8(input[input_ind]);
    uchar8 bitmask = {1<<7, 1<<6, 1<<5, 1<<4, 1<<3, 1<<2, 1<<1, 1};
    val &= bitmask;
    int8 mask = ~ convert_int8(val);
    mask += 1;
    mask = ~mask;
    mask &= 1 << 31;
    rhoind |= mask;
    temp[worker_id] = rhoind;
    mem_fence(CLK_LOCAL_MEM_FENCE);

    __local int *tempScalar = (__local int *)temp;
    // Worker i sums rho bin i and atomically updates global bins
    int localRho = worker_id;
    while (localRho < numrho && localRho + minrho < nbins) {
        int binCount = 0;
        for (int i = num_workers*8 - 1; i >= 0; i--)
            if (tempScalar[i] == localRho)
                binCount++;
        if (binCount != 0)
            atomic_add(&bins[nbins * theta_num + minrho + localRho], binCount);
        localRho += num_workers;
    }
}
