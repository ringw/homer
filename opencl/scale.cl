#define X (0)
#define Y (1)

__kernel void scale_image(__global const uchar *input,
                          float scale,
                          uint inputWidth, uint inputHeight,
                          __global uchar *output) {
    uint x = get_global_id(X);
    uint y = get_global_id(Y);
    uint input_y = convert_uint_rtn(y * scale);
    input_y = y;

    uchar result = 0;
    for (uint bit = 0; bit < 8; bit++) {
        uint bit_x = x * 8 + bit;

        uint input_x = convert_uint_rtn(bit_x * scale);
        input_x = x;
        if (input_x < inputWidth*8 && input_y < inputHeight) {
            uchar input_byte = input[input_x/8 + input_y * inputWidth];
            int input_bit = input_x % 8;
            result |= (((input_byte >> (7 - input_bit)) & 1) << (7 - bit));
        }
    }

    output[x + y * get_global_size(X)] = result;
}
