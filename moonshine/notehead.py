from .opencl import *
import numpy as np

int4 = cl.tools.get_or_register_dtype('int4')
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

// Choose random points from the array for the first point in the pair,
// with replacement.
__kernel void point1_random_choice(__global const int2 *points,
                                   uint num_points,
                                   __global volatile ulong *seed,
                                   __local volatile ulong *local_seed,
                                   __global int4 *choices) {
    uint local_id = get_local_id(0);
    ulong rand;
    if (local_id == 0) {
        // Must get local seed for all workers
        *local_seed = rand = atom_split_rand(seed);
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    if (local_id != 0) {
        rand = atom_split_rand_l(local_seed);
    }
    rand = rand_next(rand);
    uint point_ind = convert_uint_rtn(rand_val(rand) * num_points);
    choices[get_global_id(0)].s01 = points[point_ind];
}

// For each first point in the array, choose a second point within the bounds
// relative to the first point, or fill the second point with (-1, -1).
#define MAXITER 1024
__kernel void point2_random_choice(__global const int2 *points,
                                   uint num_points,
                                   __global int4 *pairs,
                                   int4 bounds, // x0, y0, width, height
                                   __global volatile ulong *seed,
                                   __local volatile ulong *local_seed) {
    uint local_id = get_local_id(0);
    ulong rand;
    if (local_id == 0) {
        // Must get local seed for all workers
        *local_seed = rand = atom_split_rand(seed);
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    if (local_id != 0) {
        rand = atom_split_rand_l(local_seed);
    }

    int4 pair = pairs[get_global_id(0)];
    pair.s23 = (int2)(-1, -1);
    int4 absolute_bounds = bounds;
    absolute_bounds.xy += pair.s01;
    for (int i = 0; i < MAXITER; i++) {
        rand = rand_next(rand);
        int2 point = points[convert_uint_rtn(num_points * rand_val(rand))];
        if (absolute_bounds.x <= point.x
            && point.x < absolute_bounds.x + absolute_bounds.s2
            && absolute_bounds.y <= point.y
            && point.y < absolute_bounds.y + absolute_bounds.s3) {
            pair.s23 = point;
            break;
        }
    }
    pairs[get_global_id(0)] = pair;
}

#define AXIS_MAX 64
#define AXIS_THRESH 16
__kernel void hough_ellipse_center(__global const int4 *border_pairs,
                                   __global const int2 *border_points,
                                   uint num_points,
                                   uint min_dist, uint max_dist,
                                   __global volatile ulong *rand,
                                   __local volatile ulong *local_rand,
                                   __local volatile uint *axis_acc,
                                   __global uchar *ellipses) {
    uint pair_ind = get_global_id(0);
    int4 pair = border_pairs[pair_ind];
    int2 p0 = pair.s01;
    int2 p1 = pair.s23;
    if (p0.x < 0 || p1.x < 0)
        return;
    float2 v01 = convert_float2(p0 - p1);
    float d01 = dot(v01, v01);

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

    if (local_id == 0) {
        for (int i = 0; i < AXIS_MAX; i++)
            axis_acc[i] = 0;
    }
    mem_fence(CLK_LOCAL_MEM_FENCE);

    for (int iter = 0; iter < 256; iter++) {
        worker_seed = rand_next(worker_seed);
        uint p2num = convert_uint_rtn(rand_val(worker_seed) * num_points);
        int2 p2 = border_points[p2num];
        float2 v02 = convert_float2(p0 - p2);
        float2 v12 = convert_float2(p1 - p2);
        float d02 = dot(v02, v02);
        float d12 = dot(v12, v12);
        if (! (min_dist * min_dist <= d02 && d02 < max_dist * max_dist
                && min_dist * min_dist <= d12 && d12 < max_dist * max_dist
                && d12 < d02 && d01 < d02))
            continue;
        float2 center = convert_float2(p0 + p1) * 0.5;
        float asquared = d01 / 4.0;
        // Distance from center to 3rd point
        float2 p2f = convert_float2(p2);
        float dsquared = dot(center - p2f, center - p2f);
        float cosSquaredTau = pow(asquared + dsquared - d12, 2) / (4 * asquared * dsquared);
        uint minor_axis = convert_uint_rtn(sqrt(asquared * dsquared * (1 - cosSquaredTau)
                                        / (asquared - dsquared * cosSquaredTau)));
        if (0 < minor_axis && minor_axis < AXIS_MAX)
            atomic_inc(&axis_acc[minor_axis]);
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    
    if (local_id == 0) {
        uchar max_axis = 0;
        uint max_count = 0;
        for (int i = 0; i < AXIS_MAX; i++) {
            uint axis_count = axis_acc[i];
            if (axis_count > max_count && axis_count >= AXIS_THRESH) {
                max_axis = i;
                max_count = axis_count;
            }
        }
        ellipses[get_global_id(0)] = max_axis;
    }
}
""").build()
prg.point1_random_choice.set_scalar_arg_dtypes([
    None, np.uint32, None, None, None
])
prg.point2_random_choice.set_scalar_arg_dtypes([
    None, np.uint32, None, int4, None, None
])
prg.hough_ellipse_center.set_scalar_arg_dtypes([
    None, None, np.uint32, np.uint32, np.uint32, None, None, None, None
])

from moonshine import filter, bitimage
NUMPAIRS = 1024 * 8
seed = cla.to_device(q,
        np.random.randint(0, 2**32, 2).astype(np.uint32).view(np.uint64))
def detect_ellipses(page, img=None):
    if img is None:
        img = page.img
    img_nostaff = filter.remove_staff(page, img)
    pixels = bitimage.pixel_where(bitimage.border(img_nostaff))
    pairs = cla.empty(q, (NUMPAIRS, 4), np.uint32)
    e=prg.point1_random_choice(q, (NUMPAIRS,), (16,),
                                pixels.data, np.uint32(pixels.shape[0]),
                                seed.data,
                                cl.LocalMemory(8),
                                pairs.data)
    e.wait()
    print (e.profile.end - e.profile.start) / 10.0 ** 9
    min_radius = page.staff_space / 2
    e=prg.point2_random_choice(q, (NUMPAIRS,), (16,),
                                pixels.data, np.uint32(pixels.shape[0]),
                                pairs.data,
                                (np.array([page.staff_space,
                                           -page.staff_space,
                                           page.staff_space, page.staff_space/2],
                                           np.int32)
                                   .view(int4)[0]),
                                seed.data,
                                cl.LocalMemory(8))
    e.wait()
    print (e.profile.end - e.profile.start) / 10.0 ** 9
    ellipses = cla.zeros(q, (pairs.shape[0],), np.uint8)
    return prg.hough_ellipse_center(q, (NUMPAIRS, 256),
                            (1, 256),
                            pairs.data,
                            pixels.data,
                            np.uint32(pixels.shape[0]),
                            np.uint32(5), np.uint32(min_radius*3),
                            seed.data,
                            cl.LocalMemory(8),
                            cl.LocalMemory(64 * 4),
                            ellipses.data), pairs, ellipses

if __name__ == "__main__":
    # Run test
    from . import image, page, measure
    p = page.Page(image.read_pages('samples/sonata.png')[0])
    p.process()
    m = measure.get_measure(p, 11, 3)
    e, pairs, ell = detect_ellipses(p, m)
    e.wait()
    print (e.profile.end - e.profile.start) / 10.0**9
    import scipy.stats
    from pylab import *
    E = ell.get()
    axis = scipy.stats.mode(E[E != 0])[0][0]
    print "axis:", axis
    imshow(np.unpackbits(m.get()).reshape((m.shape[0],-1)))
    for x0,y0,x1,y1 in pairs.get()[E == axis].astype(int):
        xc = (x0 + x1)/2.0
        yc = (y0 + y1)/2.0
        a = sqrt((x1 - x0)**2 + (y1 - y0)**2)/2
        t0 = -np.arctan2(y1 - y0, x1 - x0)
        ts = np.linspace(0, 2*pi, 100)
        e_x = a * cos(ts)
        e_y = axis * sin(ts)
        plot(xc + e_x * cos(t0) + e_y * -sin(t0),
             yc + e_y * cos(t0) + e_x * sin(t0), 'g')
    show()
