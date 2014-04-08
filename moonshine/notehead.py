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
inline ulong atom_split_rand_l(__local volatile ulong *rand) {
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
    uint patch_min = MAX(0, (int)(4 - y0));
    uint image_min = MAX(0, (int)(y0 - 4));
    uint patch_max = 8;
    uint image_max = MIN((int)image_h, (int)(8 + y0 - 4));
    uint patch_y, image_y;
    for (patch_y = patch_min, image_y = image_min;
         patch_y < patch_max && image_y < image_max;
         patch_y++, image_y++) {
        patch.a[patch_y] = image[image_x + image_w * image_y];
    }
    if (x0 % 8 < 4) {
        uchar shift = 4 - (x0 % 8);
        patch.v >>= shift;
        if (image_x >= 1) {
            for (patch_y = patch_min, image_y = image_min;
                 patch_y < patch_max && image_y < image_max;
                 patch_y++, image_y++) {
                patch.a[patch_y] |= image[image_x-1 + image_w * image_y]
                                        << 8 - shift;
            }
        }
    }
    else {
        uchar shift = (x0 % 8) - 4;
        patch.v <<= shift;
        if (image_x < image_w - 1 && shift != 0) {
            for (patch_y = patch_min, image_y = image_min;
                 patch_y < patch_max && image_y < image_max;
                 patch_y++, image_y++) {
                patch.a[patch_y] |= image[image_x+1 + image_w * image_y]
                                        >> 8 - shift;
            }
        }
    }
    return patch.v;
}

__kernel void test_patch(__global const uchar *image,
                         uint image_w, uint image_h,
                         uint x0, uint y0,
                         __global uchar8 *output) {
    *output = get_patch(image, image_w, image_h, x0, y0);
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
    if ((border_pixel.s4 & 0x08) == 0)
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

__kernel void test_tangent(__global const uchar *image,
                           uint image_w, uint image_h,
                           uint x0, uint y0,
                           __global float *output) {
    *output = patch_tangent_angle(image, image_w, image_h, x0, y0);
}

#define MAXITER 10
__kernel void hough_ellipse_center(__global const uchar *image,
                                   uint pixres, uint search_dist,
                                   __global volatile ulong *rand,
                                   __local volatile ulong *local_rand,
                                   __local float2 *local_centers,
                                   __global volatile uint *accumulator) {
    uint x0 = get_global_id(0),
         y0 = get_global_id(1),
         w = get_global_size(0)/8,
         h = get_global_size(1);
    
    uint local_id = get_local_id(0) + get_local_size(0) * get_local_id(1);
    uint num_workers = get_local_size(0) * get_local_size(1);
    ulong worker_seed;
    if (local_id == 0) {
        // Must get local seed for all workers
        *local_rand = worker_seed = atom_split_rand(rand);
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    if (local_id != 0) {
        worker_seed = atom_split_rand_l(local_rand);
    }

    float t0 = patch_tangent_angle(image, w, h, x0, y0);
    if (!isfinite(t0)) return;
    uint x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float t1 = 0, t2 = 0;

    uint xmin = MAX(0, (int)(x0 - search_dist)),
         xmax = MIN(w*8, (int)(x0 + search_dist)),
         ymin = MAX(0, (int)(y0 - search_dist)),
         ymax = MIN(h, (int)(y0 + search_dist));
    float2 center = (float2)(-1, -1);
    for (int iter = 0; iter < MAXITER; iter++) {
        worker_seed = rand_next(worker_seed);
        uint x2 = convert_uint_rtn(rand_val(worker_seed)
                                      * (xmax - xmin) + xmin);
        worker_seed = rand_next(worker_seed);
        uint y2 = convert_uint_rtn(rand_val(worker_seed)
                                      * (ymax - ymin) + ymin);
        if ((x2 == x0 && y2 == y0) || (x2 == x1 && y2 == y1))
            continue;
        float t2 = patch_tangent_angle(image, w, h, x2, y2);
        if (isfinite(t2)) {
            if (x1 == 0 && y1 == 0) {
                x1 = x2;
                y1 = y2;
                t1 = t2;
            }
            else {
                // Convert tangent lines to homogenous coordinates:
                // aX + bY + cW = 0 ==> L(a,b,c) . P(X,Y,W) = 0
                // Vectors in R^3 must be stored as float4
                float3 L0 = (float3)(sin(t0),
                                     -cos(t0),
                                     y0 * cos(t0) - x0 * sin(t0));
                float3 L1 = (float3)(sin(t1),
                                     -cos(t1),
                                     y1 * cos(t1) - x1 * sin(t1));
                float3 L2 = (float3)(sin(t2),
                                     -cos(t2),
                                     y2 * cos(t2) - x2 * sin(t2));
                // Find intersection of tangent lines T01 and T12
                float3 T01 = cross(L0, L1), T12 = cross(L1, L2);
                // Calculate midpoints between points 0/1 and 1/2
                float3 M01 = (float3)(x0 + x1, y0 + y1, 2.0);
                float3 M12 = (float3)(x1 + x2, y1 + y2, 2.0);
                // The center of the ellipse is the intersection of the lines
                // T01M01 and T12M12
                float3 T01M01 = cross(T01, M01);
                float3 T12M12 = cross(T12, M12);
                float3 center_h = cross(T01M01, T12M12);
                if (fabs(center_h.z) > 1e-5) {
                    center = center_h.xy / center_h.z;
                    float2 dist = center - (float2)(x0, y0);
                    if (dot(dist, dist) > search_dist * search_dist)
                        center = (float2)-1;
                }
                break;
            }
        }
    }
    local_centers[local_id] = center;
    mem_fence(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        // Atomically update accumulator
        for (uint i = 0; i < 64; i++) {
            float2 center = local_centers[i];
            if (0 <= center.x && center.x < w*8
                && 0 <= center.y && center.y < h) {
                uint acc_x = convert_uint_rtn(center.x / pixres);
                uint acc_y = convert_uint_rtn(center.y / pixres);
                uint acc_w = w * 8 / pixres;
                if (acc_x > 0 && acc_y > 0)
                    atom_inc(&accumulator[acc_x + acc_w * acc_y]);
            }
        }
    }
}
""").build()
prg.test_patch.set_scalar_arg_dtypes([
    None, np.uint32, np.uint32, np.uint32, np.uint32, None
])
prg.test_tangent.set_scalar_arg_dtypes([
    None, np.uint32, np.uint32, np.uint32, np.uint32, None
])
prg.hough_ellipse_center.set_scalar_arg_dtypes([
    None, np.uint32, np.uint32, None, None, None, None
])

if __name__ == "__main__":
    # Run test
    from . import image, page, rotate
    p = page.Page(image.read_pages('samples/sonata.png')[0])
    rotate.rotate(p)
#    patch = cla.zeros(q, (8,), np.uint8)
#    TEST_X=1076
#    TEST_Y=752
#    prg.test_patch(q, (1,), (1,),
#                      p.img.data,
#                      np.uint32(p.img.shape[1]),
#                      np.uint32(p.img.shape[0]),
#                      np.uint32(TEST_X), np.uint32(TEST_Y),
#                      patch.data).wait()
#    print np.unpackbits(patch.get()).reshape((8,8))
#    tangent = cla.zeros(q, (1,), np.float32)
#    prg.test_tangent(q, (1,), (1,),
#                      p.img.data,
#                      np.uint32(p.img.shape[1]),
#                      np.uint32(p.img.shape[0]),
#                      np.uint32(TEST_X), np.uint32(TEST_Y),
#                      tangent.data).wait()
#    print tangent[0]
    seed = cla.to_device(q,
               np.random.randint(0, 2**32, 2).astype(np.uint32).view(np.uint64))
    acc = cla.zeros(q, (1024, 1024), np.uint32)
    for i in xrange(5):
        prg.hough_ellipse_center(q, (p.img.shape[1] * 8, p.img.shape[0]), (8, 8),
                                    p.img.data,
                                    np.uint32(4),
                                    np.uint32(30),
                                    seed.data,
                                    cl.LocalMemory(4),
                                    cl.LocalMemory(64 * 8),
                                    acc.data).wait()
    A = acc.get()
    print (A > 0).sum()
    import pylab
    pylab.ylim([p.byteimg.shape[0], 0])
    pylab.xlim([0, p.byteimg.shape[1]])
    extent = (0,4096,4096,0)
    pylab.imshow(p.byteimg, extent=extent, cmap='binary')
    Aalpha = np.zeros((1024, 1024, 4), dtype=np.uint8)
    Aalpha[..., 0] = 255 # red
    Aalpha[..., 3] = np.uint8(np.float32(A) / A.max() * 255)
    pylab.imshow(Aalpha, extent=extent)
    pylab.show()
