#define X (0)
#define Y (1)
__kernel void scaled_bitmap_to_int_array(__global const uchar *input,
                          float scale,
                          int inputWidth, int inputHeight,
                          int zeroVal, int oneVal,
                          __global int *output) {
    int x = get_global_id(X);
    int y = get_global_id(Y);

    // Search a square area of the input, if any pixels are on set the
    // output pixel to oneVal
    int input_x0 = convert_int_rtn(x * scale),
         input_x1 = convert_int_rtn((x+1) * scale),
         input_y0 = convert_int_rtn(y * scale),
         input_y1 = convert_int_rtn((y+1) * scale);
    int output_val = zeroVal;
    for (int input_x = input_x0; input_x < input_x1; input_x++)
        for (int input_y = input_y0; input_y < input_y1; input_y++)
            if (input_x < inputWidth*8 && input_y < inputHeight) {
                uchar input_byte = input[input_x/8 + input_y * inputWidth];
                int input_bit = input_x % 8;
                if ((input_byte >> (7 - input_bit)) & 1)
                    output_val = oneVal;
            }

    output[x + y * get_global_size(X)] = output_val;
}
