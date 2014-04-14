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

#define MAXITER 50
__kernel void hough_ellipse_center(__global const int2 *border_inds,
                                   uint num_points,
                                   uint min_dist, uint max_dist,
                                   __global volatile ulong *rand,
                                   __local volatile ulong *local_rand,
                                   __local volatile uint *axis_acc,
                                   __global uchar *ellipses) {
    if (get_global_id(0) >= num_points
        || get_global_id(1) >= num_points)
        return;
    uint pair_ind = get_global_id(0) + num_points * get_global_id(1);
    int2 p0 = border_inds[get_global_id(0)];
    int2 p1 = border_inds[get_global_id(1)];
    float2 v01 = convert_float2(p0 - p1);
    float d01 = dot(v01, v01);
    if (min_dist * min_dist <= d01 && d01 < max_dist * max_dist) {
        ellipses[pair_ind] = 0;
        return;
    }
    
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

    uchar minor_axis = 0;
    for (int iter = 0; iter < MAXITER; iter++) {
        worker_seed = rand_next(worker_seed);
        uint p2num = convert_uint_rtn(rand_val(worker_seed) * num_points);
        int2 p2 = border_inds[p2num];
        float2 v02 = convert_float2(p0 - p2);
        float2 v12 = convert_float2(p1 - p2);
        float d02 = dot(v02, v02);
        float d12 = dot(v12, v12);
        if (! (min_dist * min_dist <= d02 && d02 < max_dist * max_dist
                && min_dist * min_dist <= d12 && d12 < max_dist * max_dist))
            continue;
        float2 center = convert_float2(p0 + p1) * 0.5;
        float asquared = d01 / 4.0;
        // Distance from center to 3rd point
        float2 p2f = convert_float2(p2);
        float dsquared = dot(center - p2f, center - p2f);
        float cosSquaredTau = pow(asquared + dsquared - d12, 2) / (4 * asquared * dsquared);
        minor_axis = convert_uchar_rtn(asquared * dsquared * (1 - cosSquaredTau)
                                        / (asquared - dsquared * cosSquaredTau));
    }
}
""").build()
prg.hough_ellipse_center.set_scalar_arg_dtypes([
    None, np.uint32, np.uint32, np.uint32, None, None, None, None
])

from moonshine import filter, bitimage
def detect_ellipses(page, img=None):
    if img is None:
        img = page.img
    img_nostaff = filter.remove_staff(page, img)
    pixels = bitimage.pixel_where(bitimage.border(img_nostaff))
    ellipses = cla.zeros(q, (pixels.shape[0],), np.uint8)
    seed = cla.to_device(q,
            np.random.randint(0, 2**32, 2).astype(np.uint32).view(np.uint64))
    wg_size = -(-int(pixels.shape[0]) & -32)
    return prg.hough_ellipse_center(q, (wg_size, wg_size),
                            (16, 16),
                            pixels.data,
                            np.uint32(512),
                            np.uint32(5), np.uint32(30),
                            seed.data,
                            cl.LocalMemory(8),
                            cl.LocalMemory(1),
                            ellipses.data)

if __name__ == "__main__":
    # Run test
    from . import image, page, measure
    p = page.Page(image.read_pages('samples/sonata.png')[0])
    p.process()
    m = measure.get_measure(p, 11, 3)
    e = detect_ellipses(p, m)
    e.wait()
    print (e.profile.end - e.profile.start) / 10.0**9
