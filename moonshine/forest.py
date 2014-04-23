from .opencl import *
from . import bitimage
import cPickle

prg = cl.Program(cx, """
#define BLOCK_SIZE 17
#define NUM_CLASSES 8

__kernel void run_forest(__global const uchar *image,
                         __local char *local_patch,
                         __global const int *node_feature,
                         __global const int *node_children,
                         __local volatile int *class_accumulator,
                         __global uchar *pixel_class /* NOT bit-packed */) {
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
        int feature_x = pixel_x - patch_x0
                        + (feature % BLOCK_SIZE) - BLOCK_SIZE / 2;
        int feature_y = pixel_y - patch_y0
                        + (feature / BLOCK_SIZE) - BLOCK_SIZE / 2;
        int feature_val = local_patch[feature_x + patch_w * feature_y];

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
        atomic_inc(&class_accumulator[class + NUM_CLASSES * (
                        get_local_id(0) + get_local_size(0) * get_local_id(1))]);
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
        pixel_class[pixel_x + image_w * pixel_y] = best_class;//class_accumulator[1 + NUM_CLASSES * (
                    //get_local_id(0) + get_local_size(0) * get_local_id(1))];
    }
}

""").build()

class Forest:
    num_trees = None
    features = None
    children = None

def load_forest(path):
    classifier = cPickle.load(open(path, 'rb'))
    num_trees = len(classifier.estimators_)
    tree_size = [len(e.tree_.feature) for e in classifier.estimators_]
    forest_size = sum(tree_size)
    features = cla.empty(q, forest_size, np.int32)
    children = cla.empty(q, (forest_size, 2), np.int32)
    # Map each root to the first n indices which each worker starts on,
    # then concatenate all remaining nodes from each tree
    nonroot_start_ind = num_trees + \
        np.cumsum([0] + [s - 1 for s in tree_size[:-1]])
    for i, tree in enumerate(classifier.estimators_):
        features[i] = tree.tree_.feature[0]
        start_ind = nonroot_start_ind[i]
        children[i, 0] = (start_ind + tree.tree_.children_left[0] - 1
                          if tree.tree_.children_left[0] > 0
                          else -1)
        children[i, 1] = (start_ind + tree.tree_.children_right[0] - 1
                          if tree.tree_.children_right[0] > 0
                          else -1)
        new_features = tree.tree_.feature[1:].astype(np.int32)
        # Extract values from the leaves
        new_features[new_features < 0] = \
            -1 - np.argmax(tree.tree_.value[tree.tree_.feature < 0], axis=-1)
        features[start_ind : start_ind + tree_size[i]-1] = new_features
        orig_children = np.c_[tree.tree_.children_left[1:],
                              tree.tree_.children_right[1:]]
        new_children = (start_ind - 1 + orig_children).astype(np.int32)
        new_children[orig_children < 0] = -1
        children[start_ind : start_ind + tree_size[i]-1, :] = new_children
    f = Forest()
    f.num_trees = num_trees
    f.features = features
    f.children = children
    return f

def predict(forest, bitimg):
    img_classes = cla.zeros(q, (bitimg.shape[0], bitimg.shape[1] * 8),
                               np.uint8)
    local_size=8
    e=prg.run_forest(q, img_classes.shape[::-1] + (forest.num_trees,),
                      (local_size, local_size, forest.num_trees),
                      bitimg.data,
                      cl.LocalMemory((local_size + 17) * (local_size + 17)),
                      forest.features.data,
                      forest.children.data,
                      cl.LocalMemory(4 * local_size * local_size * 8),
                      img_classes.data)
    e.wait()
    print (e.profile.end-e.profile.start) / 10.0 ** 9
    return img_classes

def scale_img(page):
    assert page.staff_dist >= 8
    scale = 8.0 / page.staff_dist
    scaled_img = bitimage.scale(page.img, scale, align=64)
    return scaled_img, scale

def classify(forest, page):
    scaled_img, scale = scale_img(page)
    return predict(forest, scaled_img)

if __name__ == '__main__':
    import moonshine
    page, = moonshine.open('samples/chopin.pdf')
    page.process()
    f = load_forest('classifier.pkl')
    # Test for cycles
    children = f.children.get()
    tree = cPickle.load(open('classifier.pkl','rb')).estimators_[0].tree_
    children_ = np.c_[tree.children_left, tree.children_right]
    leaves = []
    def recurse(i):
        print i
        if children[i,0] < 0:
            leaves.append(i)
            return
        else:
            recurse(children[i,0])
            recurse(children[i,1])
    #recurse(0)
    classes = classify(f, page)
    from pylab import *
    imshow(classes.get())
    colorbar()
    show()
