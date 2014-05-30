__kernel void transpose(__global const uchar *image,
                        __global uchar *image_T) {
    // Get coordinates in output image
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint w = get_global_size(0);
    uint h = get_global_size(1);

    uchar output_byte = 0;
    for (uint b = 0; b < 8; b++) {
        uchar input_byte = image[y/8 + h/8 * (x*8 + b)];
        input_byte >>= 7 - (y % 8);
        input_byte &= 0x1;
        output_byte |= input_byte << (7 - b);
    }
    image_T[x + w * y] = output_byte;
}

__kernel void scale_image(__global const uchar *input,
                          float scale,
                          uint inputWidth, uint inputHeight,
                          __global uchar *output) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint input_y = convert_uint_rtn(y / scale);

    uchar result = 0;
    for (uint bit = 0; bit < 8; bit++) {
        uint bit_x = x * 8 + bit;

        uint input_x = convert_uint_rtn(bit_x / scale);
        if (input_x < inputWidth*8 && input_y < inputHeight) {
            uchar input_byte = input[input_x/8 + input_y * inputWidth];
            int input_bit = input_x % 8;
            result |= (((input_byte >> (7 - input_bit)) & 1) << (7 - bit));
        }
    }

    output[x + y * get_global_size(0)] = result;
}

__kernel void erode(__global const uchar *image,
                    __global uchar *output_image) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint w = get_global_size(0);
    uint h = get_global_size(1);

    uchar input_byte = image[x + w * y];
    uchar output_byte = input_byte;
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

__kernel void dilate(__global const uchar *image,
                     __global uchar *output_image) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint w = get_global_size(0);
    uint h = get_global_size(1);

    uchar input_byte = image[x + w * y];
    uchar output_byte = input_byte;
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

__kernel void border(__global const uchar *image,
                     __global uchar *output_image) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint w = get_global_size(0);
    uint h = get_global_size(1);

    uchar input_byte = image[x + w * y];
    uchar erosion = input_byte;
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
