from .opencl import *
from . import bitimage
import cPickle

prg = build_program('forest')

class Forest:
    num_trees = None
    features = None
    children = None
    classes = None

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
    f.classes = classifier.classes_
    return f
classifier = load_forest('classifier.pkl')

def predict(forest, bitimg):
    img_classes = cla.zeros(q, (bitimg.shape[0], bitimg.shape[1] * 8),
                               np.uint8)
    local_w=8
    local_h=4
    e=prg.run_forest(q, img_classes.shape[::-1] + (forest.num_trees,),
                      (local_w, local_h, forest.num_trees),
                      bitimg.data,
                      cl.LocalMemory((local_w + 35) * (local_h + 35)),
                      forest.features.data,
                      forest.children.data,
                      cl.LocalMemory(4 * local_w * local_h * forest.num_trees),
                      img_classes.data)
    e.wait()
    return img_classes

def scale_img(page, img=None):
    assert page.staff_dist >= 8
    if img is None:
        img = page.img
    scale = 8.0 / page.staff_dist
    scaled_img = bitimage.scale(img, scale, align=64)
    return scaled_img, scale

def classify(forest, page, img=None):
    scaled_img, scale = scale_img(page, img=img)
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
    scaled_img, scale = scale_img(page)
    classes = predict(f, scaled_img)
#    from pylab import *
#    bw = np.zeros((scaled_img.shape[0], scaled_img.shape[1]*8, 3), np.uint8)
#    bw[:] = np.where(np.unpackbits(scaled_img.get())
#                       .reshape((scaled_img.shape[0], -1)), 0, 255)[:,:,None]
#    imshow(bw)
#    C = classes.get()
#    imshow(C, alpha=0.5)
#    show()
    import moonshine.components
    c,b,s = moonshine.components.get_components(classes)
