// Connected component analysis

inline void uf_union(GLOBAL_MEM volatile int *tree,
                     int n1, int n2) {
    int parent, child; // child is root of tree being merged with parent
    do {
        int p1 = n1, p2 = n2; // Parent of current node
        do {
            n1 = p1;
            n2 = p2;
            p1 = tree[n1];
            p2 = tree[n2];
        } while (! ((p1 == n1) && (p2 == n2)));

        // Make the parent always have the lower pixel position
        // The tree is probably very unbalanced but it's fast enough
        parent = n1 < n2 ? n1 : n2;
        child  = n1 < n2 ? n2 : n1;
    } while (atomic_cmpxchg(&tree[child], child, parent) != child);
}

/* Create connected component trees for each connected component with a
 * distinct nonzero class.
 */
KERNEL void init_component_tree(GLOBAL_MEM const UCHAR *classes,
                                GLOBAL_MEM volatile int *pixel_tree) {
    // Initialize a nonzero pixel with its own id
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int id = x + w * y;
    UCHAR class = classes[id];
    if (class == 0)
        pixel_tree[id] = 0;
    else
        pixel_tree[id] = id;
}

KERNEL void build_component_tree(GLOBAL_MEM const UCHAR *classes,
                                 GLOBAL_MEM volatile int *pixel_tree) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);
    int id = x + w * y;
    UCHAR class = classes[id];
    if (class == 0)
        return;
    if (y > 0) {
        if (x > 0 && classes[(x-1) + w * (y-1)] == class)
            uf_union(pixel_tree, (x-1) + w * (y-1), id);
        if (classes[x + w * (y-1)] == class)
            uf_union(pixel_tree, x + w * (y-1), id);
        if (x < w && classes[(x+1) + w * (y-1)] == class)
            uf_union(pixel_tree, (x+1) + w * (y-1), id);
    }
    if (x > 0 && classes[(x-1) + w * y] == class)
        uf_union(pixel_tree, (x-1) + w * y, id);
}

KERNEL void count_components(GLOBAL_MEM int *pixel_tree,
                             GLOBAL_MEM volatile int *num_components) {
    int id = get_global_id(0) + get_global_size(0) * get_global_id(1);
    // Replace root of a tree with negative unique ID for the tree
    // Root points to itself
    // XXX: we need to handle the top left point correctly
    if (id != 0 && pixel_tree[id] == id) {
        pixel_tree[id] = -1 - atomic_inc(num_components);
    }
}

KERNEL void init_component_bounds(GLOBAL_MEM int *component_bounds,
                                  const int2 image_size) {
    int component_num = get_global_id(0);
    component_bounds[component_num * 4 + 0] = image_size.x;
    component_bounds[component_num * 4 + 1] = 0;
    component_bounds[component_num * 4 + 2] = image_size.y;
    component_bounds[component_num * 4 + 3] = 0;
}

KERNEL void component_info(GLOBAL_MEM const UCHAR *classes,
                           GLOBAL_MEM const int *pixel_tree,
                           GLOBAL_MEM UCHAR *component_classes,
                           GLOBAL_MEM volatile int *component_bounds,
                           GLOBAL_MEM volatile int *component_sums) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);
    int id = x + w * y;
    if (pixel_tree[id] == 0)
        return;

    int parent = id;
    int root;
    // XXX: terminate search after 1024 steps
    // We should figure out why we are looping
    int k = 1024;
    do {
        root = parent;
        parent = pixel_tree[root];
    } while (parent >= 0 && k--);
    int component_num = -1 - parent;

    if (root == id) {
        // We started at the root, so this worker updates the component class
        component_classes[component_num] = classes[id];
    }

    atomic_min(&component_bounds[component_num * 4 + 0], x);
    atomic_max(&component_bounds[component_num * 4 + 1], x);
    atomic_min(&component_bounds[component_num * 4 + 2], y);
    atomic_max(&component_bounds[component_num * 4 + 3], y);
    atomic_inc(&component_sums[component_num]);
}
