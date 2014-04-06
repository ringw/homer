# Notehead detection, Python reference implementation
import numpy as np

def patch(bitimg, x, y):
    """ Return an 8x8 patch of bits centered around (x, y) """
    # Store the patch locally centered on (x, y)
    patch_size = 8
    x_offset = patch_size // 2
    y_offset = patch_size // 2
    patch = np.zeros((patch_size, patch_size // 8), np.uint8)
    # Load the bytes x belongs to, some y values may be zero (outside image)
    img_ymin = max(0, y - y_offset)
    img_ymax = min(bitimg.shape[0], patch_size + y - y_offset)
    ymin = max(0, y_offset - y)
    ymax = min(patch_size, y_offset + bitimg.shape[0] - y)
    print img_ymin, img_ymax, ymin, ymax
    patch[ymin:ymax] = bitimg[img_ymin:img_ymax, x // 8, None]
    if x % 8 < x_offset:
        patch >>= x_offset - (x % 8)
        if x >= 8:
            patch |= (bitimg[img_ymin:img_ymax, (x // 8) - 1, None]
                        << 8 - (x_offset - (x % 8)))
    else:
        patch <<= (x % 8) - x_offset
        if x // 8 < bitimg.shape[1] - 1 and x % 8 != x_offset:
            patch |= (bitimg[img_ymin:img_ymax, (x // 8) + 1, None]
                        >> 8 - ((x % 8) - x_offset))
    return patch

def border_tangent(bitimg, x, y):
    """ Find dark pixels adjacent to a light pixel in an
        8x8 area around (x, y), fit a line to the
        coordinates of those pixels, and return the angle of the
        line to the horizontal. If (x, y) is not actually a border
        pixel, then return NaN.
    """
    img_patch = patch(bitimg, x, y)
    # Calculate binary erosion
    erosion = img_patch.copy()
    erosion &= img_patch >> 1
    erosion &= img_patch << 1
    erosion[:-1] &= img_patch[1:] & (img_patch[1:] >> 1) & (img_patch[1:] << 1)
    erosion[1:] &= (img_patch[:-1] & (img_patch[:-1] >> 1)
                                   & (img_patch[:-1] << 1))
    # Must not erode from edges, otherwise border will exist on the edges
    # if it was already dark
    erosion[[0, -1]] = 0xFF
    erosion |= 0x81
    border_pixel = img_patch & ~ erosion

    if border_pixel[4, 0] & 0x10 == 0:
        # (x, y) is not a border pixel
        return np.nan
    # To calculate the linear regression, we need to sum x, y, x^2, x*y, y^2
    # for all border pixel coordinates
    # Use lookup tables for sum_x and sum_xx. SUM_1_LUT counts number of on bits
    SUM_1_LUT = np.zeros(256, np.float32)
    for b in xrange(8):
        SUM_1_LUT[(np.arange(256) >> b) & 1 != 0] += 1
    SUM_X_LUT = np.zeros(256, np.float32)
    for b in xrange(8):
        SUM_X_LUT[(np.arange(256) >> b) & 1 != 0] += 7 - b
    SUM_XX_LUT = np.zeros(256, np.float32)
    for b in xrange(8):
        SUM_XX_LUT[(np.arange(256) >> b) & 1 != 0] += (7 - b) * (7 - b)

    num_points = np.sum(SUM_1_LUT[border_pixel])
    if num_points == 1:
        return np.nan
    sum_x = np.sum(SUM_X_LUT[border_pixel])
    sum_y = np.sum(np.arange(8)[:, None] * SUM_1_LUT[border_pixel])
    sum_xy = np.sum(np.arange(8)[:, None] * SUM_X_LUT[border_pixel])
    sum_xx = np.sum(SUM_XX_LUT[border_pixel])

    #patch_y, patch_x = np.where(np.unpackbits(border_pixel).reshape((8,8)))
    #assert sum_x == np.sum(patch_x)
    #assert sum_y == np.sum(patch_y)
    #assert sum_xy == np.sum(patch_x * patch_y)
    #assert sum_xx == np.sum(patch_x ** 2)

    # Calculate slope as mn/md for input to atan2
    mn = sum_x * sum_y - num_points * sum_xy
    md = sum_x ** 2 - num_points * sum_xx
    return np.arctan2(mn, md)
