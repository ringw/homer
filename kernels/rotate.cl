KERNEL void rotate_image(GLOBAL_MEM const UCHAR *input,
                           float cos_theta, float sin_theta,
                           GLOBAL_MEM UCHAR *output) {
    UCHAR result = 0;
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);
    for (int bit = 0; bit < 8; bit++) {
        int bit_x = global_x * 8 + bit;
        int input_x = convert_int_rtn(bit_x * cos_theta + global_y * sin_theta);
        int input_y = convert_int_rtn(-bit_x * sin_theta + global_y * cos_theta);
        if (0 <= input_x && input_x < get_global_size(0)*8
            && 0 <= input_y && input_y < get_global_size(1)) {
            UCHAR input_byte = input[input_x/8 + input_y * get_global_size(0)];
            int input_bit = input_x % 8;
            result |= (((input_byte >> (7 - input_bit)) & 1) << (7 - bit));
        }
    }

    output[global_x + global_y * get_global_size(0)] = result;
}
