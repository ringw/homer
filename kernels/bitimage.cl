KERNEL void transpose(GLOBAL_MEM const UCHAR *image,
                        GLOBAL_MEM UCHAR *image_T) {
    // Get coordinates in output image
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

    UCHAR output_byte = 0;
    for (int b = 0; b < 8; b++) {
        UCHAR input_byte = image[y/8 + h/8 * (x*8 + b)];
        input_byte >>= 7 - (y % 8);
        input_byte &= 0x1;
        output_byte |= input_byte << (7 - b);
    }
    image_T[x + w * y] = output_byte;
}

KERNEL void scale_image(GLOBAL_MEM const UCHAR *input,
                          float scale,
                          int inputWidth, int inputHeight,
                          GLOBAL_MEM UCHAR *output) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int input_y = convert_int_rtn(y / scale);

    UCHAR result = 0;
    for (int bit = 0; bit < 8; bit++) {
        int bit_x = x * 8 + bit;

        int input_x = convert_int_rtn(bit_x / scale);
        if (input_x < inputWidth*8 && input_y < inputHeight) {
            UCHAR input_byte = input[input_x/8 + input_y * inputWidth];
            int input_bit = input_x % 8;
            result |= (((input_byte >> (7 - input_bit)) & 1) << (7 - bit));
        }
    }

    output[x + y * get_global_size(0)] = result;
}

KERNEL void erode(GLOBAL_MEM const UCHAR *image,
                    GLOBAL_MEM UCHAR *output_image) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

    UCHAR input_byte = image[x + w * y];
    UCHAR output_byte = input_byte;
    // Erode inner bits from left and right
    output_byte &= 0x80 | (input_byte >> 1);
    output_byte &= 0x01 | (input_byte << 1);

    // Erode MSB from left and LSB from right
    if (x > 0)
        output_byte &= 0x7F | (image[x-1 + w * y] << 7);
    if (x < w - 1)
        output_byte &= 0xFE | (image[x+1 + w * y] >> 7);

    // Erode from above and below
    if (y > 0)
        output_byte &= image[x + w * (y-1)];
    if (y < h - 1)
        output_byte &= image[x + w * (y+1)];
    
    output_image[x + w * y] = output_byte;
}

KERNEL void dilate(GLOBAL_MEM const UCHAR *image,
                     GLOBAL_MEM UCHAR *output_image) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

    UCHAR input_byte = image[x + w * y];
    UCHAR output_byte = input_byte;
    // Dilate inner bits from left and right
    output_byte |= input_byte >> 1;
    output_byte |= input_byte << 1;

    // Dilate MSB from left and LSB from right
    if (x > 0)
        output_byte |= image[x-1 + w * y] << 7;
    if (x < w - 1)
        output_byte |= image[x+1 + w * y] >> 7;

    // Dilate from above and below
    if (y > 0)
        output_byte |= image[x + w * (y-1)];
    if (y < h - 1)
        output_byte |= image[x + w * (y+1)];
    
    output_image[x + w * y] = output_byte;
}

KERNEL void border(GLOBAL_MEM const UCHAR *image,
                     GLOBAL_MEM UCHAR *output_image) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

    UCHAR input_byte = image[x + w * y];
    UCHAR erosion = input_byte;
    // Erode inner bits from left and right
    erosion &= 0x80 | (input_byte >> 1);
    erosion &= 0x01 | (input_byte << 1);

    // Erode MSB from left and LSB from right
    if (x > 0)
        erosion &= 0x7F | (image[x-1 + w * y] << 7);
    if (x < w - 1)
        erosion &= 0xFE | (image[x+1 + w * y] >> 7);

    // Erode from above and below
    if (y > 0)
        erosion &= image[x + w * (y-1)];
    if (y < h - 1)
        erosion &= image[x + w * (y+1)];
    
    output_image[x + w * y] = input_byte & ~erosion;
}

KERNEL void copy_bits_complex64(GLOBAL_MEM const UCHAR *bitimage,
                                  int x0, int y0, int in_w,
                                  GLOBAL_MEM float2 *complex_image) {
    int out_x = get_global_id(0);
    int out_y = get_global_id(1);
    int out_w = get_global_size(0);
    int in_x = (x0 + out_x) / 8;
    int in_bit = (x0 + out_x) % 8;
    int in_y = y0 + out_y;

    UCHAR in_byte = bitimage[in_x + in_w * in_y];
    float2 out;
    out.x = (in_byte >> (7 - in_bit)) & 0x01;
    out.y = 0;
    complex_image[out_x + out_w * out_y] = out;
}
