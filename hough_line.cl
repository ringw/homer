__kernel void hough_line(__global const uchar *input,
                       __global const float *tan_theta,
                       int rhores, int nbins,
                       __local volatile int *temp,
                       __global volatile int *bins) {
    // Get top left x and y of block
    int blockW = get_local_size(0);
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

    uchar val = input[input_ind];
    if (val) {
        int blockX = get_local_id(0);
        int blockY = get_local_id(1);
        int rhoind = convert_int_rtn((-tt * blockX + blockY) / rhores);
        temp[worker_id] = rhoind;
    }
    else {
        temp[worker_id] = -1;
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);

    // Worker i sums rho bin i and atomically updates global bins
    int globalRho = minrho + worker_id;
    if (worker_id < numrho && globalRho < nbins) {
        int binCount = 0;
        for (int i = num_workers - 1; i >= 0; i--)
            if (temp[i] == worker_id)
                binCount++;
        if (binCount != 0)
            atomic_add(&bins[nbins * theta_num + globalRho], binCount);
    }
}
