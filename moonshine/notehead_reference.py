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
    patch[ymin:ymax] = bitimg[img_ymin:img_ymax, x // 8, None]
    if x % 8 < x_offset:
        patch >>= x_offset - (x % 8)
        if x >= 8:
            patch[ymin:ymax] |= (bitimg[img_ymin:img_ymax, (x // 8) - 1, None]
                                    << 8 - (x_offset - (x % 8)))
    else:
        patch <<= (x % 8) - x_offset
        if x // 8 < bitimg.shape[1] - 1 and x % 8 != x_offset:
            patch[ymin:ymax] |= (bitimg[img_ymin:img_ymax, (x // 8) + 1, None]
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

    # Calculate slope as mn/md for input to atan2
    mn = sum_x * sum_y - num_points * sum_xy
    md = sum_x ** 2 - num_points * sum_xx
    return np.arctan2(mn, md)

MAXITER = 100
def hough_ellipse_fit(bitimg, x0, y0, maxdist=30):
    """ If (x0, y0) is a border point, choose two other border points within
        maxdist and find the candidate ellipse center.
    """
    t0 = border_tangent(bitimg, x0, y0)
    if np.isnan(t0):
        return None
    x1 = y1 = t1 = x2 = y2 = t2 = None
    for i in xrange(MAXITER):
        x = np.random.randint(max(0, x0 - maxdist),
                              min(bitimg.shape[1]*8, x0 + maxdist))
        y = np.random.randint(max(0, y0 - maxdist),
                              min(bitimg.shape[0], y0 + maxdist))
        if x == x0 and y == y0:
            continue
        t = border_tangent(bitimg, x, y)
        if not np.isnan(t):
            if x1 is None:
                (x1, y1, t1) = (x, y, t)
            elif x != x1 or y != y1:
                (x2, y2, t2) = (x, y, t)
        if x1 is not None and x2 is not None:
            # Convert tangent lines to homogenous coordinates:
            # aX + bY + cW = 0 ==> L(a,b,c) . P(X,Y,W) = 0
            L0 = [np.sin(t0), -np.cos(t0), -x0 * np.sin(t0) + y0 * np.cos(t0)]
            L1 = [np.sin(t1), -np.cos(t1), -x1 * np.sin(t1) + y1 * np.cos(t1)]
            L2 = [np.sin(t2), -np.cos(t2), -x2 * np.sin(t2) + y2 * np.cos(t2)]
            # Find intersections of tangent lines T01 and T12
            T01 = np.cross(L0, L1)
            T12 = np.cross(L1, L2)
            # Calculate midpoints between original pixels
            M01 = [x0 + x1, y0 + y1, 2.0]
            M12 = [x1 + x2, y1 + y2, 2.0]
            # The center of the ellipse is the intersection of the lines
            # T01M01 and T12M12
            T01M01 = np.cross(T01, M01)
            T12M12 = np.cross(T12, M12)
            ellipse_center = np.cross(T01M01, T12M12)
            if np.abs(ellipse_center[2]) > 1e-5:
                return ellipse_center[:2] / ellipse_center[2]
            else:
                x1 = y1 = t1 = x2 = y2 = t2 = None
    return None
if __name__ == "__main__":
    from . import image, page
    p = page.Page(image.read_pages('samples/sonata.png')[0])
    centers = []
    for i in xrange(1000):
        ell = hough_ellipse_fit(p.bitimg, 1095, 747)
        if ell is not None:
            centers.append(ell)
    import pylab
    pylab.figure()
    pylab.imshow(p.byteimg)
    pylab.figure()
    pylab.plot(*(tuple(np.array(centers).T) + ('g.',)))
    pylab.show()
