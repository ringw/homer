from .opencl import *

prg = cl.Program(cx, """
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
""").build()

def transpose(img):
    assert img.shape[0] % 8 == 0
    img_T = cla.zeros(q, (img.shape[1] * 8, img.shape[0] // 8), np.uint8)
    prg.transpose(q, img_T.shape[::-1], (1, 8), img.data, img_T.data).wait()
    return img_T

def repeat_kernel(img, kernel, numiter=1):
    temp_1 = cla.zeros_like(img)
    if numiter > 1:
        temp_2 = cla.zeros_like(img)
    kernel(q, img.shape[::-1], (16, 8),
              img.data, temp_1.data).wait()
    for i in xrange(1, numiter, 2):
        kernel(q, img.shape[::-1], (16, 8),
                  temp_1.data, temp_2.data).wait()
        kernel(q, img.shape[::-1], (16, 8),
                  temp_2.data, temp_1.data).wait()
    if numiter % 2 == 1:
        return temp_1
    else:
        kernel(q, img.shape[::-1], (16, 8),
                  temp_1.data, temp_2.data).wait()
        return temp_2

def erode(img, numiter=1):
    return repeat_kernel(img, prg.erode, numiter)
def dilate(img, numiter=1):
    return repeat_kernel(img, prg.dilate, numiter)
def opening(img, numiter=1):
    return dilate(erode(img, numiter), numiter)
def closing(img, numiter=1):
    return erode(dilate(img, numiter), numiter)
