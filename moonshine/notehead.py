from .opencl import *
import numpy as np

prg = cl.Program(cx, """
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

#define MAX(a,b) (((a)>(b)) ? (a) : (b))
#define MIN(a,b) (((a)<(b)) ? (a) : (b))

// Random number algorithm borrowed from Haskell's random package
typedef int random_state[2];
union random_long {
    random_state r;
    ulong l;
};
inline ulong rand_next(ulong rand_long) {
    union random_long r_union;
    r_union.l = rand_long;
    random_state rand;
    rand[0] = r_union.r[0];
    rand[1] = r_union.r[1];
    int k0 = rand[0] / 53668;
    int s0 = 40014 * (rand[0] % 53668) - k0 * 12211;
    if (s0 < 0) s0 += 2147483563;
    int k1 = rand[1] / 52774;
    int s1 = 40692 * (rand[1] % 52774) - k1 * 3791;
    if (s1 < 0) s1 += 2147483399;
    random_state rand2;
    r_union.r[0] = s0;
    r_union.r[1] = s1;
    return r_union.l;
}
// Convert random state to a uniform random value on the interval [0, 1)
inline float rand_val(ulong rand) {
    union random_long r_union;
    r_union.l = rand;
    long randVal = (long)r_union.r[1] - r_union.r[0];
    randVal += 2147483562;
    randVal %= 2147483562;
    return (float)randVal / 2147483562.f;
}
inline ulong atom_rand(__global volatile ulong *rand) {
    ulong newVal, prevVal;
    do {
        prevVal = *rand;
        newVal = rand_next(prevVal);
    } while (atom_cmpxchg(rand, prevVal, newVal) != prevVal);
    return newVal;
}
inline ulong atom_rand_l(__local volatile ulong *rand) {
    ulong newVal, prevVal;
    do {
        prevVal = *rand;
        newVal = rand_next(prevVal);
    } while (atom_cmpxchg(rand, prevVal, newVal) != prevVal);
    return newVal;
}
// Split global rand with a second stream which can generate random numbers
// in parallel
// Warning: according to Haskell source, no statistical foundation for this
inline ulong2 split_rand(ulong rand) {
    ulong2 splitVal;
    union random_long ru;
    ru.l = rand;
    int s1 = ru.r[0];
    int s2 = ru.r[1];
    ru.l = rand_next(rand);
    int t1 = ru.r[0];
    int t2 = ru.r[1];
    // Update rand with (new_s1, t2)
    ru.r[0] = (s1 == 2147483562) ? 1 : s1 + 1;
    // Create a second stream (t1, new_s2)
    splitVal.s0 = ru.l;
    ru.r[0] = t1;
    ru.r[1] = (s2 == 1) ? 2147483398 : s2 - 1;
    splitVal.s1 = ru.l;
    return splitVal;
}
    
inline ulong atom_split_rand(__global volatile ulong *rand) {
    ulong newVal, prevVal, retVal;
    do {
        prevVal = *rand;
        ulong2 splitVal = split_rand(prevVal);
        newVal = splitVal.s0;
        retVal = splitVal.s1;
    } while (atom_cmpxchg(rand, prevVal, newVal) != prevVal);
    return retVal;
}

inline uchar8 get_patch(__global const uchar *image,
                        uint image_w, uint image_h,
                        uint x0, uint y0) {
    union {
        uchar8 v;
        uchar a[8];
    } patch;
    patch.v = (uchar8)(0);
    
    uint image_x = x0 / 8;
    uint patch_y, image_y;
    for (patch_y = MAX(0, 4 - y0), image_y = MAX(0, y0 - 4);
         patch_y < 8 && image_y < MAX(image_h, 8 + y0 - 4);
         patch_y++, image_y++) {
        patch.a[patch_y] = image[image_x + image_w * image_y];
    }
    return patch.v;
}

__constant float SUM_1_LUT[256] = {
    0.0,1.0,1.0,2.0,1.0,2.0,2.0,3.0,
    1.0,2.0,2.0,3.0,2.0,3.0,3.0,4.0,
    1.0,2.0,2.0,3.0,2.0,3.0,3.0,4.0,
    2.0,3.0,3.0,4.0,3.0,4.0,4.0,5.0,
    1.0,2.0,2.0,3.0,2.0,3.0,3.0,4.0,
    2.0,3.0,3.0,4.0,3.0,4.0,4.0,5.0,
    2.0,3.0,3.0,4.0,3.0,4.0,4.0,5.0,
    3.0,4.0,4.0,5.0,4.0,5.0,5.0,6.0,
    1.0,2.0,2.0,3.0,2.0,3.0,3.0,4.0,
    2.0,3.0,3.0,4.0,3.0,4.0,4.0,5.0,
    2.0,3.0,3.0,4.0,3.0,4.0,4.0,5.0,
    3.0,4.0,4.0,5.0,4.0,5.0,5.0,6.0,
    2.0,3.0,3.0,4.0,3.0,4.0,4.0,5.0,
    3.0,4.0,4.0,5.0,4.0,5.0,5.0,6.0,
    3.0,4.0,4.0,5.0,4.0,5.0,5.0,6.0,
    4.0,5.0,5.0,6.0,5.0,6.0,6.0,7.0,
    1.0,2.0,2.0,3.0,2.0,3.0,3.0,4.0,
    2.0,3.0,3.0,4.0,3.0,4.0,4.0,5.0,
    2.0,3.0,3.0,4.0,3.0,4.0,4.0,5.0,
    3.0,4.0,4.0,5.0,4.0,5.0,5.0,6.0,
    2.0,3.0,3.0,4.0,3.0,4.0,4.0,5.0,
    3.0,4.0,4.0,5.0,4.0,5.0,5.0,6.0,
    3.0,4.0,4.0,5.0,4.0,5.0,5.0,6.0,
    4.0,5.0,5.0,6.0,5.0,6.0,6.0,7.0,
    2.0,3.0,3.0,4.0,3.0,4.0,4.0,5.0,
    3.0,4.0,4.0,5.0,4.0,5.0,5.0,6.0,
    3.0,4.0,4.0,5.0,4.0,5.0,5.0,6.0,
    4.0,5.0,5.0,6.0,5.0,6.0,6.0,7.0,
    3.0,4.0,4.0,5.0,4.0,5.0,5.0,6.0,
    4.0,5.0,5.0,6.0,5.0,6.0,6.0,7.0,
    4.0,5.0,5.0,6.0,5.0,6.0,6.0,7.0,
    5.0,6.0,6.0,7.0,6.0,7.0,7.0,8.0,
};
__constant float SUM_X_LUT[256] = {
    0.0,7.0,6.0,13.0,5.0,12.0,11.0,18.0,
    4.0,11.0,10.0,17.0,9.0,16.0,15.0,22.0,
    3.0,10.0,9.0,16.0,8.0,15.0,14.0,21.0,
    7.0,14.0,13.0,20.0,12.0,19.0,18.0,25.0,
    2.0,9.0,8.0,15.0,7.0,14.0,13.0,20.0,
    6.0,13.0,12.0,19.0,11.0,18.0,17.0,24.0,
    5.0,12.0,11.0,18.0,10.0,17.0,16.0,23.0,
    9.0,16.0,15.0,22.0,14.0,21.0,20.0,27.0,
    1.0,8.0,7.0,14.0,6.0,13.0,12.0,19.0,
    5.0,12.0,11.0,18.0,10.0,17.0,16.0,23.0,
    4.0,11.0,10.0,17.0,9.0,16.0,15.0,22.0,
    8.0,15.0,14.0,21.0,13.0,20.0,19.0,26.0,
    3.0,10.0,9.0,16.0,8.0,15.0,14.0,21.0,
    7.0,14.0,13.0,20.0,12.0,19.0,18.0,25.0,
    6.0,13.0,12.0,19.0,11.0,18.0,17.0,24.0,
    10.0,17.0,16.0,23.0,15.0,22.0,21.0,28.0,
    0.0,7.0,6.0,13.0,5.0,12.0,11.0,18.0,
    4.0,11.0,10.0,17.0,9.0,16.0,15.0,22.0,
    3.0,10.0,9.0,16.0,8.0,15.0,14.0,21.0,
    7.0,14.0,13.0,20.0,12.0,19.0,18.0,25.0,
    2.0,9.0,8.0,15.0,7.0,14.0,13.0,20.0,
    6.0,13.0,12.0,19.0,11.0,18.0,17.0,24.0,
    5.0,12.0,11.0,18.0,10.0,17.0,16.0,23.0,
    9.0,16.0,15.0,22.0,14.0,21.0,20.0,27.0,
    1.0,8.0,7.0,14.0,6.0,13.0,12.0,19.0,
    5.0,12.0,11.0,18.0,10.0,17.0,16.0,23.0,
    4.0,11.0,10.0,17.0,9.0,16.0,15.0,22.0,
    8.0,15.0,14.0,21.0,13.0,20.0,19.0,26.0,
    3.0,10.0,9.0,16.0,8.0,15.0,14.0,21.0,
    7.0,14.0,13.0,20.0,12.0,19.0,18.0,25.0,
    6.0,13.0,12.0,19.0,11.0,18.0,17.0,24.0,
    10.0,17.0,16.0,23.0,15.0,22.0,21.0,28.0,
};
__constant float SUM_XX_LUT[256] = {
    0.0,49.0,36.0,85.0,25.0,74.0,61.0,110.0,
    16.0,65.0,52.0,101.0,41.0,90.0,77.0,126.0,
    9.0,58.0,45.0,94.0,34.0,83.0,70.0,119.0,
    25.0,74.0,61.0,110.0,50.0,99.0,86.0,135.0,
    4.0,53.0,40.0,89.0,29.0,78.0,65.0,114.0,
    20.0,69.0,56.0,105.0,45.0,94.0,81.0,130.0,
    13.0,62.0,49.0,98.0,38.0,87.0,74.0,123.0,
    29.0,78.0,65.0,114.0,54.0,103.0,90.0,139.0,
    1.0,50.0,37.0,86.0,26.0,75.0,62.0,111.0,
    17.0,66.0,53.0,102.0,42.0,91.0,78.0,127.0,
    10.0,59.0,46.0,95.0,35.0,84.0,71.0,120.0,
    26.0,75.0,62.0,111.0,51.0,100.0,87.0,136.0,
    5.0,54.0,41.0,90.0,30.0,79.0,66.0,115.0,
    21.0,70.0,57.0,106.0,46.0,95.0,82.0,131.0,
    14.0,63.0,50.0,99.0,39.0,88.0,75.0,124.0,
    30.0,79.0,66.0,115.0,55.0,104.0,91.0,140.0,
    0.0,49.0,36.0,85.0,25.0,74.0,61.0,110.0,
    16.0,65.0,52.0,101.0,41.0,90.0,77.0,126.0,
    9.0,58.0,45.0,94.0,34.0,83.0,70.0,119.0,
    25.0,74.0,61.0,110.0,50.0,99.0,86.0,135.0,
    4.0,53.0,40.0,89.0,29.0,78.0,65.0,114.0,
    20.0,69.0,56.0,105.0,45.0,94.0,81.0,130.0,
    13.0,62.0,49.0,98.0,38.0,87.0,74.0,123.0,
    29.0,78.0,65.0,114.0,54.0,103.0,90.0,139.0,
    1.0,50.0,37.0,86.0,26.0,75.0,62.0,111.0,
    17.0,66.0,53.0,102.0,42.0,91.0,78.0,127.0,
    10.0,59.0,46.0,95.0,35.0,84.0,71.0,120.0,
    26.0,75.0,62.0,111.0,51.0,100.0,87.0,136.0,
    5.0,54.0,41.0,90.0,30.0,79.0,66.0,115.0,
    21.0,70.0,57.0,106.0,46.0,95.0,82.0,131.0,
    14.0,63.0,50.0,99.0,39.0,88.0,75.0,124.0,
    30.0,79.0,66.0,115.0,55.0,104.0,91.0,140.0,
};

inline float patch_tangent_angle(__global const uchar *image,
                                 uint image_w, uint image_h,
                                 uint x0, uint y0) {
    uchar8 patch = get_patch(image, image_w, image_h, x0, y0);
    uchar8 erosion = patch;
    erosion &= patch >> (uchar)1;
    erosion &= patch << (uchar)1;
    erosion.s0123456 &= patch.s1234567 & (patch.s1234567 << (uchar)1)
                                       & (patch.s1234567 >> (uchar)1);
    erosion.s1234567 &= patch.s0123456 & (patch.s0123456 << (uchar)1)
                                       & (patch.s0123456 >> (uchar)1);
    uchar8 border_pixel = patch & ~ erosion;

    // Patch must be centered on a border pixel
    if ((patch.s4 & 0x10) == 0)
        return NAN;
    // Store num_points, sum_x, sum_y, sum_xy, sum_xx of points along border
    // which we find, in order to use linear regression
    float num_points = 0, sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
""" + "".join([
    """num_points += SUM_1_LUT[border_pixel.s""" + str(i) + """];
    sum_x += SUM_X_LUT[border_pixel.s""" + str(i) + """];
    sum_y += """+str(i)+""".0 * SUM_1_LUT[border_pixel.s""" + str(i) + """];
    sum_xy += """+str(i)+""".0 * SUM_X_LUT[border_pixel.s""" + str(i) + """];
    sum_xx += SUM_XX_LUT[border_pixel.s""" + str(i) + "];\n"
    for i in xrange(8)
]) + """

    // Calculate slope as mn/md for input to atan2
    float mn = sum_x * sum_y - num_points * sum_xy;
    float md = sum_x * sum_x - num_points * sum_xx;
    return atan2(mn, md);
}

#define MAXITER 100
__kernel void hough_ellipse_center(__global const uchar *image,
                                   uint pixres, uint search_dist,
                                   __global volatile ulong *rand,
                                   __local volatile ulong *local_rand,
                                   __global volatile uint *accumulator) {
    uint x = get_global_id(0),
         y = get_global_id(1),
         w = get_global_size(0),
         h = get_global_size(1);

    float t0 = patch_tangent_angle(image, w, h, x, y);
    if (isnan(t0)) return;
    uint x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float t1 = 0, t2 = 0;

    uint xmin = MAX(0, x - search_dist),
         xmax = MIN(w*8, x + search_dist),
         ymin = MAX(0, y - search_dist),
         ymax = MIN(h, y + search_dist);
    for (int iter = 0; iter < MAXITER; iter++) {
        
    }
}
""").build()

