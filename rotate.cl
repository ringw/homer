__kernel void rotate_image(__global const uchar *input,
                           float sin_theta, float cos_theta,
                           /*__local uchar *temp,*/
                           __global uchar *output) {
    // Store patch of image required for this block of output
    /*int blockW = get_local_size(0)*8;
    int blockH = get_local_size(1);
    int x0 = blockW * get_group_id(0);
    int y0 = blockH * get_group_id(1);
    int x1 = x0 + blockW;
    int y1 = y0 + blockH;

    float input_x0_f = x0 * cos_theta + min(y0 * sin_theta, y1 * sin_theta);
    int input_x0 = convert_int_rtn(input_x0_f / 8.0);
    float input_x1_f = x1 * sin_theta + max(y0 * sin_theta, y1 * sin_theta);
    int input_x1 = convert_int_rtp(input_x1_f / 8.0);
    float input_y0_f = min(-x0 * sin_theta, -x1 * sin_theta) + y0 * cos_theta;
    int input_y0 = convert_int_rtn(input_y0_f);
    float input_y1_f = max(-x0 * sin_theta, -x1 * sin_theta) + y1 * cos_theta;
    int input_y1 = convert_int_rtp(input_y1_f);

    int temp_cols = input_x1 - input_x0;
    int temp_rows = input_y1 - input_y0;
    int temp_size = temp_cols * temp_rows;

    int worker_id = get_local_id(0) + get_local_id(1) * get_local_size(0);
    int num_workers = get_local_size(0) * get_local_size(1);
    for (int temp_ind = worker_id; temp_ind < temp_size;
         temp_ind += num_workers) {
        int temp_row = temp_ind / temp_cols;
        int temp_col = temp_ind % temp_cols;
        int global_row = temp_row + input_x0;
        int global_col = temp_col + input_y0;
        if (0 <= global_row && global_row < get_global_size(1)
            && 0 <= global_col && global_col < get_global_size(0))
            temp[temp_ind] =
                input[global_col + global_row * get_global_size(0)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);*/

    uchar result = 0;
    int bit = 0;
    int global_x = get_global_id(0) * 8 + bit;
    int global_y = get_global_id(1);
    int input_x = convert_int_rtn(global_x * cos_theta + global_y * sin_theta);
    int input_y = convert_int_rtn(-global_x * sin_theta + global_y * cos_theta);
    if (0 <= input_x && input_x < get_global_size(0)*8
        && 0 <= input_y && input_y < get_global_size(1)) {
        uchar input_byte = input[input_x/8 + input_y * get_global_size(0)];
        result = input_byte; // XXX
    }

    output[global_x + global_y * get_global_size(0)] = result;
}
