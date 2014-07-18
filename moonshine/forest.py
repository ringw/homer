from .opencl import *
from . import bitimage
import cPickle

prg = build_program('forest')

class GPUForest:
    num_trees = None
    features = None
    threshold = None
    children = None
    classes = None
    def __init__(self, forest):
        self.num_trees = forest['num_trees']
        self.features = cla.to_device(q, forest['features'])
        self.threshold = cla.to_device(q, forest['threshold'])
        self.children = cla.to_device(q, forest['children'])
        self.classes = forest['classes']
cpu_classifier = cPickle.load(open('classifier.pkl'))
classifier = GPUForest(cpu_classifier)

NUM_CLASSES = 32
BLOCK_SIZE = 35
PROJ_SIZE = 15
def predict_cpu(forest, patch, num_results=1):
    assert patch.shape == (BLOCK_SIZE, BLOCK_SIZE)
    nodes = np.arange(forest['num_trees'])
    predictions = np.zeros(NUM_CLASSES, int)
    vproj = patch.sum(0)
    hproj = patch.sum(1)
    while len(nodes):
        # Add leaves' class to hte accumulator and remove them from nodes
        feats = forest['features'][nodes]
        leaves = feats < 0
        leaf_class = -feats[leaves] - 1
        bins = np.bincount(leaf_class)
        predictions[:len(bins)] += bins
        nodes = nodes[~leaves]

        feats = forest['features'][nodes]
        seen = np.zeros_like(feats, bool)
        which_child = np.zeros_like(feats, bool)
        pixel_feat = feats < BLOCK_SIZE * BLOCK_SIZE
        which_child[pixel_feat] = patch.ravel()[feats[pixel_feat]]
        seen |= pixel_feat

        vproj_feat = (~seen) & (pixel_feat < BLOCK_SIZE*BLOCK_SIZE + PROJ_SIZE)
        which_child[vproj_feat] = (vproj[feats[vproj_feat]-BLOCK_SIZE*BLOCK_SIZE]
            >= forest['threshold'][nodes[vproj_feat]]).astype(int)
        seen |= vproj_feat

        hproj_feat = (~seen)&(pixel_feat < BLOCK_SIZE*BLOCK_SIZE + PROJ_SIZE*2)
        which_child[hproj_feat] = (hproj[feats[hproj_feat]-BLOCK_SIZE*BLOCK_SIZE-PROJ_SIZE]
            >= forest['threshold'][nodes[hproj_feat]]).astype(int)
        seen |= hproj_feat

        assert seen.all()
        nodes = forest['children'][nodes, which_child.astype(int)]
    if num_results == 1:
        return np.argmax(predictions)
    else:
        top_results = np.argsort(-predictions)
        return top_results[:num_results]

def predict(forest, bitimg, get_classes=True, num_workers=32):
    img_classes = cla.zeros(q, (bitimg.shape[0], bitimg.shape[1] * 8),
                               np.uint8)
    local_w=8
    local_h=4
    e=prg.run_forest(q, img_classes.shape[::-1] + (num_workers,),
                      (local_w, local_h, num_workers),
                      bitimg.data,
                      cl.LocalMemory((local_w + 35) * (local_h + 35)),
                      np.int32(forest.num_trees),
                      forest.features.data,
                      forest.children.data,
                      forest.threshold.data,
                      cl.LocalMemory(4 * local_w * local_h * NUM_CLASSES),
                      img_classes.data,
                      np.int32(get_classes))
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
