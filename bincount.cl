/* Bin multiple independent samples along 0th axis in 2D array
 * (1st and 2nd axes). temp must store nbins ints and bins has size
 * get_global_size(0) * nbins
 */
__kernel void bincount(__global const short *input,
                       int nbins,
                       __local volatile int *temp,
                       __global volatile int *bins) {
    int num_workers = get_local_size(1) * get_local_size(2);
    int bins_per_worker = nbins / num_workers;
    int worker_id = get_local_id(1) * get_local_size(2) + get_local_id(2);
    
    for (int i = 0; i < bins_per_worker; i++) {
        temp[worker_id + i*num_workers] = 0;
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    int input_ind = get_global_id(1) * get_global_size(2) + get_global_id(2);

    short val = input[input_ind];
    if (0 <= val && val < nbins)
        atomic_inc(&temp[val]);
    mem_fence(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < bins_per_worker; i++) {
        int bin = worker_id + i*num_workers;
        atomic_add(&bins[bin], temp[bin]);
    }
}
