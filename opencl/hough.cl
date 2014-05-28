/*
 * hough_line: Hough transform for lines
 * (image_w, imageH) are the dimensions in bytes of the bit-packed image
 * theta is angle to the horizontal
 * bins are shape (len(theta), len(rho))
 * global size should be (len(rho) * num_workers, len(theta))
 * local size may be (num_workers, 1), multiple workers will sum pixels in parallel
 */
__kernel void hough_line(__global const uchar *image,
                         uint image_w, uint image_h,
                         uint rhores,
                         __global const float *cos_thetas,
                         __global const float *sin_thetas,
                         __local float *worker_sums,
                         __global float *bins) {
    uint rho = get_group_id(0);
    uint num_rho = get_num_groups(0);
    uint theta = get_global_id(1);

    float rho_val = rho * rhores;
    float cos_theta = cos_thetas[theta];
    float sin_theta = sin_thetas[theta];

    // Multiple workers help sum up the same rho
    uint worker_id = get_local_id(0);
    uint num_workers = get_local_size(0);

    float worker_sum = 0.f;
    // Sum each x-byte position. As an approximation, assume if the left
    // corner is parameterized as (rho, theta) then we can sum up the whole byte
    for (int x = 0; x < image_w; x += num_workers) {
        float x_left_val = x * 8;
        float y_val = (rho_val - x_left_val * sin_theta) / cos_theta;
        int y = convert_int_rtn(y_val);

        if (0 <= x && x < image_w && 0 <= y && y < image_h) {
            uchar byte = image[x + image_w * y];
            uint8 bits = (uint8)byte;
            bits >>= (uint8)(7, 6, 5, 4, 3, 2, 1, 0);
            bits &= (uint8)(0x1);
            // Sum using float dot product (faster)
            float8 fbits = convert_float8(bits);
            float4 one = (float4)(1.f);
            worker_sum += dot(fbits.s0123, one);
            worker_sum += dot(fbits.s4567, one);
        }
    }

    if (num_workers > 1) {
        worker_sums[worker_id] = worker_sum;
        mem_fence(CLK_LOCAL_MEM_FENCE);
        if (worker_id == 0) {
            // Sum all partial sums into global bin
            float global_sum = 0.f;
            for (int i = 0; i < num_workers; i++)
                global_sum += worker_sums[i];
            bins[rho + num_rho * theta] = global_sum;
        }
    }
    else {
        // We are the only worker
        bins[rho + num_rho * theta] = worker_sum;
    }
}

/*
 * hough_lineseg: extract longest line segment from each Hough peak
 * Index of rho and theta on 0th axis, local size of 1
 * For each line, output (x0, x1, y0, y1) in segments
 */

__kernel void hough_lineseg(__global const uchar *image,
                            uint image_w, uint image_h,
                            __global const uint *rho_bins,
                            uint rhores,
                            __global const float *cos_thetas,
                            __global const float *sin_thetas,
                            __local uchar *segment_pixels,
                            uint max_gap,
                            __global int4 *segments) {
    uint line_id = get_group_id(0);
    uint worker_id = get_local_id(0);
    uint num_workers = get_local_size(0);

    uint rho_bin = rho_bins[line_id];
    float cos_theta = cos_thetas[line_id];
    float sin_theta = sin_thetas[line_id];

    // Iterate over pixels comprising the line
    // XXX this assumes rhores = 1
    float x = 0.f, x0 = x;
    float y = rho_bin * rhores / cos_theta, y0 = y;
    int num_pixels = 0;
    while (0 <= x && x < image_w*8 && 0 <= y && y < image_h) {
        // Iterate over xslice, yslice inside this slice with height rhores
        // perpendicular to actual line
        float xslice = x, yslice = y;
        uchar is_segment = 0;
        for (int s = 0; s < rhores; s++) {
            int xind = convert_int_rtn(xslice);
            int yind = convert_int_rtn(yslice);
            uchar byte = image[xind/8 + image_w * yind];
            is_segment |= (byte >> (7 - (xind % 8))) & 0x1;

            // Move along normal line
            xslice -= sin_theta;
            yslice += cos_theta;
        }
        segment_pixels[num_pixels] = is_segment;

        x += cos_theta;
        y -= sin_theta;
        num_pixels++;
    }

    // Prefix sum of contiguous pixels, keeping track of a maximum
    int max_length = 0;
    int max_end = -1;
    int cur_length = 0;
    int cur_skip = 0;

    for (int i = 1; i < num_pixels; i++) {
        if (segment_pixels[i]) {
            if (cur_skip < max_gap) {
                // Add any skip to the length
                cur_length += cur_skip;
                cur_skip = 0;

                cur_length++;
                if (cur_length > max_length) {
                    max_length = cur_length;
                    max_end = i;
                }
            }
            else {
                // Start a new segment
                cur_skip = 0;
                cur_length = 1;
            }
        }
        else
            cur_skip++;
    }

    if (max_length > 0) {
        int max_start = max_end - max_length + 1;
        segments[line_id] = (int4)(x0 + cos_theta * max_start,
                                   x0 + cos_theta * max_end,
                                   y0 - sin_theta * max_start,
                                   y0 - sin_theta * max_end);
    }
}

#define MIN(a,b) (((a)<(b)) ? (a) : (b))
#define MAX(a,b) (((a)>(b)) ? (a) : (b))

// Sort hough line segments before using.
// Set can_join to 1 if the segment should be considered part of the same
// staff or barline as the previous segment. This is prefix summed
// to get unique ids for each staff.
__kernel void can_join_segments(__global const int4 *segments,
                                __global int *can_join,
                                int threshold) {
    uint i = get_global_id(0);
    if (i == 0)
        can_join[i] = 0;
    else
        can_join[i] = abs(MIN(segments[i].s2, segments[i].s3)
                            - MAX(segments[i-1].s2, segments[i-1].s3))
                            > threshold
                    ? 1 : 0;
}

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
// Assign longest segment for each label atomically.
// Everything should be a ushort4 to allow atomic operations on long values.
__kernel void assign_segments(__global const ushort4 *segments,
                              __global const int *labels,
                              __global volatile ushort4 *longest) {
    uint i = get_global_id(0);
    ushort4 seg = segments[i];
    float4 segf = convert_float4(seg);
    // Get length of segment from dot product
    float dx = dot(segf, (float4)(-1, 1, 0, 0));
    float dy = dot(segf, (float4)(0, 0, -1, 1));
    float seg_length = dot((float2)(dx, dy), (float2)(dx, dy));
    int label = labels[i];
    union {
        ushort4 s;
        ulong l;
    } seg_u, old_u;
    seg_u.s = seg;
    do {
        ushort4 longest_seg = longest[label];
        segf = convert_float4(longest_seg);
        dx = dot(segf, (float4)(-1, 1, 0, 0));
        dy = dot(segf, (float4)(0, 0, -1, 1));
        float longest_length = dot((float2)(dx, dy), (float2)(dx, dy));
        if (longest_length >= seg_length)
            break;
        old_u.s = longest_seg;
    } while (atom_cmpxchg((__global volatile ulong *)&longest[label],
                          old_u.l, seg_u.l) != old_u.l);
}

// See if we can extend each initial Hough segment with another
// almost-parallel segment to increase its length.
__kernel void extend_lines(__global const ushort4 *segments,
                           __global const ushort4 *initial_segment,
                           __global const int *labels,
                           __global volatile ushort4 *extension) {
    uint i = get_global_id(0);
    union {
        ushort4 s;
        ulong l;
    } seg_u, old_u;
    seg_u.s = segments[i];
    float4 seg = convert_float4(seg_u.s);
    int label = labels[i];
    old_u.s = initial_segment[i];
}
