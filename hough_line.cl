__kernel void hough_line(__global const uchar *input,
                       __global const float *tan_theta,
                       int rhores, int numrho,
                       __local volatile int *temp,
                       __global volatile int *bins) {
    int theta_num = get_global_id(0);
    int num_workers = get_local_size(1) * get_local_size(2);
    int bins_per_worker = numrho / num_workers;
    int worker_id = get_local_id(1) * get_local_size(2) + get_local_id(2);
    
    for (int i = 0; i < bins_per_worker; i++) {
        temp[worker_id + i*num_workers] = 0;
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    int input_ind = get_global_id(1) * get_global_size(2) + get_global_id(2);

    uchar val = input[input_ind];
    if (val) {
        int y = get_global_id(1);
        int x = get_global_id(2);
        float rho = -tan_theta[theta_num]*x + y;
        rho /= rhores;
        int rhoind = convert_int(rho);
        if (0 <= rhoind && rhoind < numrho)
            atomic_inc(&temp[rhoind]);
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < bins_per_worker; i++) {
        int bin = worker_id + i*num_workers;
        atomic_add(&bins[theta_num * numrho + bin], temp[bin]);
    }
}
