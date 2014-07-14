#define BLOCK_SIZE 35
#define PROJ_SIZE 15
#define NUM_CLASSES 32

__kernel void run_forest(__global const uchar *image,
                         __local char *local_patch,
                         __global const int *node_feature,
                         __global const int *node_children,
                         __global const int *node_threshold,
                         __local volatile int *class_accumulator,
                         __global uchar *pixel_class, /* NOT bit-packed */
                         int return_class) {
    int pixel_x = get_global_id(0);
    int pixel_y = get_global_id(1);
    int image_w = get_global_size(0);
    int image_h = get_global_size(1);
    int block_x0 = get_group_id(0) * get_local_size(0);
    int block_y0 = get_group_id(1) * get_local_size(1);
    int patch_x0 = block_x0 - BLOCK_SIZE / 2;
    int patch_y0 = block_y0 - BLOCK_SIZE / 2;
    int patch_w = get_local_size(0) + BLOCK_SIZE;
    int patch_h = get_local_size(1) + BLOCK_SIZE;

    int worker_id = get_local_id(0) + get_local_size(0) * (
        get_local_id(1) + get_local_size(1) * get_global_id(2));
    int num_workers = get_local_size(0) * get_local_size(1) * get_local_size(2);
    for (int i = worker_id; i < patch_w * patch_h; i += num_workers) {
        int x = patch_x0 + i % patch_w;
        int y = patch_y0 + i / patch_w;
        if (0 <= x && x < image_w && 0 <= y && y < image_h) {
            int byte_x = x / 8;
            uchar byte = image[byte_x + image_w / 8 * y];
            local_patch[i] = (byte >> (7 - (x % 8))) & 0x1;
        }
        else {
            local_patch[i] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int node = get_global_id(2);
    int feature = node_feature[node];
    int c=0;
    while (feature >= 0) {
        int threshold = node_threshold[node];
        int feature_val;
        if (feature < BLOCK_SIZE * BLOCK_SIZE) {
            int feature_x = pixel_x - patch_x0
                            + (feature % BLOCK_SIZE) - BLOCK_SIZE / 2;
            int feature_y = pixel_y - patch_y0
                            + (feature / BLOCK_SIZE) - BLOCK_SIZE / 2;
            feature_val = local_patch[feature_x + patch_w * feature_y];
        }
        else if (feature < BLOCK_SIZE * BLOCK_SIZE + PROJ_SIZE) {
            // Compare vertical projection to threshold
            int feature_x = pixel_x - patch_x0 - PROJ_SIZE/2
                                + (feature - BLOCK_SIZE*BLOCK_SIZE);
            int proj = 0;
            feature_val = 0;
            for (int dy = -PROJ_SIZE/2; dy <= PROJ_SIZE/2; dy++) {
                int feature_y = pixel_y - patch_y0 + dy;
                proj += local_patch[feature_x + patch_w * feature_y];
                if (proj >= threshold) {
                    feature_val = 1;
                    break;
                }
            }
        }
        else {
            // Compare horizontal projection to threshold
            int feature_y = pixel_y - patch_y0 - PROJ_SIZE/2
                                + (feature - BLOCK_SIZE*BLOCK_SIZE - PROJ_SIZE);
            int proj = 0;
            feature_val = 0;
            for (int dx = -PROJ_SIZE/2; dx <= PROJ_SIZE/2; dx++) {
                int feature_x = pixel_x - patch_x0 + dx;
                proj += local_patch[feature_x + patch_w * feature_y];
                if (proj >= threshold) {
                    feature_val = 1;
                    break;
                }
            }
        }
        node = node_children[node * 2 + (feature_val ? 1 : 0)];
        feature = node_feature[node];
    }
    int class = -feature - 1;

    // Clear accumulator array
    for (int i = worker_id;
         i < NUM_CLASSES * get_local_size(0) * get_local_size(1);
         i += num_workers)
        class_accumulator[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Accumulate our class
    if (0 <= class && class < NUM_CLASSES)
        atomic_inc(&class_accumulator[class + NUM_CLASSES * (get_local_id(0)
                            + get_local_size(0) * get_local_id(1))]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Find class outputted by majority of decision trees
    if (get_local_id(2) == 0) {
        int best_class = 0;
        int class_count = 0;
        for (int i = 0; i < NUM_CLASSES; i++) {
            int this_count = class_accumulator[i + NUM_CLASSES * (
                    get_local_id(0) + get_local_size(0) * get_local_id(1))];
            if (this_count > class_count) {
                best_class = i;
                class_count = this_count;
            }
        }
        pixel_class[pixel_x + image_w * pixel_y] = return_class ? best_class
                                                    : class_count;
    }
}
