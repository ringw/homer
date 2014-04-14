from .opencl import *
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel
from pyopencl.scan import GenericScanKernel

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
def border(img):
    return repeat_kernel(img, prg.border, 1)

# Create LUT and stringify into preamble of map kernel
LUT = np.zeros(256, np.uint32)
for b in xrange(8):
    LUT[(np.arange(256) & (1 << b)) != 0] += 1
strLUT = "constant uint LUT[256] = {" + ",".join(map(str, LUT)) + "};\n"
sum_byte_count = ReductionKernel(cx, np.uint32, neutral="0",
                    reduce_expr="a+b", map_expr="LUT[bytes[i]]",
                    arguments="__global unsigned char *bytes",
                    preamble=strLUT)
def count_bits(img):
    return sum_byte_count(img).get().item()

pixel_inds = GenericScanKernel(cx, np.uint32,
                    arguments="__global unsigned char *bytes, "
                              "unsigned int image_w, "
                              "__global unsigned int *pixels",
                    # Keep count of pixels we have stored so far
                    input_expr="LUT[bytes[i]]",
                    scan_expr="a+b", neutral="0",
                    output_statement="""
                        uint pix_ind = prev_item;
                        uchar byte = bytes[i];
                        uchar mask = 0x80U;
                        uint x = (i % image_w) * 8;
                        uint y = i / image_w;
                        for (int b = 7; b >= 0; b--) {
                            if (byte & mask) {
                                vstore2((uint2)(x, y), pix_ind++, pixels);
                            }
                            mask >>= 1;
                            x++;
                        }
                    """,
                    preamble=strLUT)
def pixel_where(img):
    num_on = int(count_bits(img))
    inds = cla.empty(q, (num_on, 2), np.uint32)
    pixel_inds(img.reshape((img.shape[0] * img.shape[1],)),
               np.uint32(img.shape[1]),
               inds)
    return inds
