from .gpu import *
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel
from pyopencl.scan import GenericScanKernel

def as_bitimage(img):
    return cla.to_device(q, np.packbits(img).reshape((img.shape[0], -1)))
def as_hostimage(img):
    return np.unpackbits(img.get()).reshape((img.shape[0], -1))
def imshow(img):
    import pylab
    pylab.imshow(as_hostimage(img))

prg = build_program("bitimage")

def transpose(img):
    assert img.shape[0] % 8 == 0
    img_T = thr.empty_like(Type(np.uint8, (img.shape[1] * 8, img.shape[0] // 8)))
    img_T.fill(0)
    prg.transpose(img, img_T,
                  global_size=img_T.shape[::-1],
                  local_size=(1, 8))
    return img_T

def scale(img, scale_x, scale_y=None, align=8):
    """ Scale and round output dimension up to alignment """
    if scale_y is None:
        scale_y = scale_x
    out_h = -(-int(img.shape[0] * scale_y) & -align)
    out_w = -(-int(img.shape[1] * scale_x) & -align)
    out_img = cla.zeros(q, (out_h, out_w), np.uint8)
    prg.scale_image(img,
                    np.float32(scale_x), np.float32(scale_y),
                    np.int32(img.shape[1]), np.int32(img.shape[0]),
                    out_img,
                    global_size=out_img.shape[::-1],
                    local_size=(8, 8))
    return out_img

def repeat_kernel(img, kernel, numiter=1):
    temp_1 = thr.empty_like(img)
    temp_1.fill(0)
    if numiter > 1:
        temp_2 = thr.empty_like(img)
        temp_2.fill(0)
    kernel(img, temp_1, global_size=img.shape[::-1], local_size=(16, 8))
    for i in xrange(1, numiter, 2):
        kernel(temp_1, temp_2, global_size=img.shape[::-1], local_size=(16, 8))
        kernel(temp_2, temp_1, global_size=img.shape[::-1], local_size=(16, 8))
    if numiter % 2 == 1:
        return temp_1
    else:
        kernel(temp_1, temp_2, global_size=img.shape[::-1], local_size=(16, 8))
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
#LUT = np.zeros(256, np.int32)
#for b in xrange(8):
#    LUT[(np.arange(256) & (1 << b)) != 0] += 1
#strLUT = "constant int LUT[256] = {" + ",".join(map(str, LUT)) + "};\n"
#sum_byte_count = ReductionKernel(cx, np.int32, neutral="0",
#                    reduce_expr="a+b", map_expr="LUT[bytes[i]]",
#                    arguments="__global unsigned char *bytes",
#                    preamble=strLUT)
#def count_bits(img):
#    return sum_byte_count(img).get().item()
#
#pixel_inds = GenericScanKernel(cx, np.int32,
#                    arguments="__global unsigned char *bytes, "
#                              "int image_w, "
#                              "__global int *pixels",
#                    # Keep count of pixels we have stored so far
#                    input_expr="LUT[bytes[i]]",
#                    scan_expr="a+b", neutral="0",
#                    output_statement="""
#                        int pix_ind = prev_item;
#                        uchar byte = bytes[i];
#                        uchar mask = 0x80U;
#                        int x = (i % image_w) * 8;
#                        int y = i / image_w;
#                        for (int b = 7; b >= 0; b--) {
#                            if (byte & mask) {
#                                vstore2((int2)(x, y), pix_ind++, pixels);
#                            }
#                            mask >>= 1;
#                            x++;
#                        }
#                    """,
#                    preamble=strLUT)
#def pixel_where(img):
#    num_on = int(count_bits(img))
#    inds = cla.empty(q, (num_on, 2), np.int32)
#    pixel_inds(img.reshape((img.shape[0] * img.shape[1],)),
#               np.int32(img.shape[1]),
#               inds)
#    return inds
