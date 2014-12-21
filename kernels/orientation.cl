KERNEL void transform(GLOBAL_MEM UCHAR *output,
                      GLOBAL_MEM const UCHAR *input,
                      int input_w, int input_h,
                      GLOBAL_MEM const float *M) {
    UCHAR result = 0;
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);
    for (int bit = 0; bit < 8; bit++) {
        int bit_x = global_x * 8 + bit;
        int input_x = convert_int_rtn(bit_x * M[0] + global_y * M[1] + M[2]);
        int input_y = convert_int_rtn(bit_x * M[3] + global_y * M[4] + M[5]);

        if (0 <= input_x && input_x < input_w*8
            && 0 <= input_y && input_y < input_h) {
            UCHAR input_byte = input[input_x/8 + input_w * input_y];
            int input_bit = input_x % 8;
            result |= (((input_byte >> (7 - input_bit)) & 1) << (7 - bit));
        }
    }

    output[global_x + global_y * get_global_size(0)] = result;
}
