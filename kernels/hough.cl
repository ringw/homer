/*
 * hough_line: Hough transform for lines
 * (image_w, imageH) are the dimensions in bytes of the bit-packed image
 * theta is angle to the horizontal
 * bins are shape (len(theta), len(rho))
 * global size should be (len(rho) * num_workers, len(theta))
 * local size may be (num_workers, 1), multiple workers will sum pixels in parallel
 */
__kernel void hough_line(__global const uchar *image,
                         int image_w, int image_h,
                         int rhores,
                         __global const float *cos_thetas,
                         __global const float *sin_thetas,
                         __local float *worker_sums,
                         __global float *bins) {
    int rho = get_group_id(0);
    int num_rho = get_num_groups(0);
    int theta = get_global_id(1);

    float rho_val = rho * rhores;
    float cos_theta = cos_thetas[theta];
    float sin_theta = sin_thetas[theta];

    // Multiple workers help sum up the same rho
    int worker_id = get_local_id(0);
    int num_workers = get_local_size(0);

    float worker_sum = 0.f;
    // Sum each x-byte position. As an approximation, assume if the left
    // corner is parameterized as (rho, theta) then we can sum up the whole byte
    for (int x = 0; x < image_w; x += num_workers) {
        float x_left_val = x * 8;
        float y_val = (rho_val - x_left_val * sin_theta) / cos_theta;
        int y = convert_int_rtn(y_val);

        if (0 <= x && x < image_w && 0 <= y && y < image_h) {
            uchar byte = image[x + image_w * y];
            int8 bits = (int8)byte;
            bits >>= (int8)(7, 6, 5, 4, 3, 2, 1, 0);
            bits &= (int8)(0x1);
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
                            int image_w, int image_h,
                            __global const int *rho_bins,
                            int rhores,
                            __global const float *cos_thetas,
                            __global const float *sin_thetas,
                            __local uchar *segment_pixels,
                            int max_gap,
                            __global int4 *segments) {
    int line_id = get_group_id(0);
    int worker_id = get_local_id(0);
    int num_workers = get_local_size(0);

    int rho_bin = rho_bins[line_id];
    float cos_theta = cos_thetas[line_id];
    float sin_theta = sin_thetas[line_id];

    // Iterate over pixels comprising the line
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

        // segment_pixels is bit-packed
        int seg_byte = num_pixels / 8;
        int seg_bit  = num_pixels % 8;
        uchar seg_char;
        if (seg_bit == 0)
            seg_char = 0; // initialize segment_pixels
        else
            seg_char = segment_pixels[seg_byte];
        seg_char |= is_segment << (7 - seg_bit);
        segment_pixels[seg_byte] = seg_char;

        x += cos_theta;
        y -= sin_theta;
        num_pixels++;
    }

    // Prefix sum of contiguous pixels, keeping track of a maximum
    int max_length = 0;
    int max_end = -1;
    int cur_length = 0;
    int cur_skip = max_gap + 1; // can't start with a skip

    for (int i = 1; i < num_pixels; i++) {
        if (segment_pixels[i/8] & (0x80U >> (i%8))) {
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
    int i = get_global_id(0);
    if (i == 0)
        can_join[i] = 0;
    else
        can_join[i] = MIN(abs(segments[i-1].s2 - segments[i].s2),
                      MIN(abs(segments[i-1].s2 - segments[i].s3),
                      MIN(abs(segments[i-1].s3 - segments[i].s2),
                          abs(segments[i-1].s3 - segments[i].s3))))
                            > threshold
                    ? 1 : 0;
}

// Assign longest segment for each label atomically.
// Best distance metric is Chebyshev distance (max(dx, dy))
// to avoid favoring highly skewed lines
// longest_inds must already be initialized to all -1s
#define CHEBYSHEV(v) MAX(abs(v.s1-v.s0), abs(v.s3-v.s2))
__kernel void assign_segments(__global const int4 *segments,
                              __global const int *labels,
                              __global volatile int *longest_inds) {
    int i = get_global_id(0);
    int label = labels[i];

    int4 seg = segments[i];
    // Get length of segment from dot product
    int seg_length = CHEBYSHEV(seg);
    int longest_seg_ind;
    do {
        longest_seg_ind = longest_inds[label];
        // If someone else set the longest segment, we need to make sure
        // ours is actually longer than theirs
        if (longest_seg_ind >= 0) {
            int4 longest_seg = segments[longest_seg_ind];
            int longest_length = CHEBYSHEV(longest_seg);
            if (longest_length >= seg_length)
                break;
        }
    } while (atomic_cmpxchg(&longest_inds[label],
                            longest_seg_ind, i) != longest_seg_ind);
}

__kernel void copy_chosen_segments(__global const int4 *segments,
                                   __global const int *longest_inds,
                                   __global int4 *chosen_segs) {
    int label_ind = get_global_id(0);
    chosen_segs[label_ind] = segments[longest_inds[label_ind]];
}

// TODO: To deal with some scores where the page is warped, we may need
// to detect 2 distinct Hough peaks and then join them in a single path.
