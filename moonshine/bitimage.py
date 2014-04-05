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
""").build()

def transpose(img):
    assert img.shape[0] % 8 == 0
    img_T = cla.zeros(q, (img.shape[1] * 8, img.shape[0] // 8), np.uint8)
    prg.transpose(q, img_T.shape[::-1], (1, 8), img.data, img_T.data).wait()
    return img_T
