import numpy as np
from sklearn.ensemble import RandomForestClassifier
import moonshine
from moonshine.preprocessing import staffsize
import scipy.misc
import scipy.ndimage
import glob
import os.path
import cPickle

from forest_config import COLOR_LABELS

PATCH_SIZE = 35
LABEL_SIZE = 5
SURROUND_SIZE = 5 # background around labeled pixels
object_size = {
    "treble_clef": 7,
    "bass_clef": 7,
    "small_treble_clef": 5,
    "small_bass_clef": 5,
    "beam": 1
}

labeled_data = glob.glob('labels/*.png')
all_patches = []
all_patch_labels = []
for labels in labeled_data:
    # patches for this file
    patches = []
    patch_labels = []

    label_img = scipy.misc.imread(labels)
    image_path = os.path.join('samples', os.path.basename(labels))
    image, = moonshine.open(image_path)
    staffsize.staffsize(image)
    assert image.staff_dist >= 8
    image_scale = 8.0 / image.staff_dist
    image = scipy.misc.imresize(image.byteimg[:image.orig_size[0],
                                              :image.orig_size[1]].astype(bool),
                                image_scale,
                                interp='nearest')
    num_label_patches = 0
    for label_type, label_name in COLOR_LABELS.iteritems():
        our_patches = []
        our_patch_labels = []
        is_label = label_img[:, :, :3] == np.array([[label_type]])
        # When scaling down, need to ensure each label pixel maps to some
        # new pixel and isn't overwritten by the background
        label_y, label_x = np.where(is_label.all(axis=-1))
        if len(label_y):
            label_y = (label_y * image_scale).astype(int)
            label_x = (label_x * image_scale).astype(int)
            scale_label = np.zeros_like(image)
            scale_label[label_y, label_x] = 1
            size = LABEL_SIZE
            if label_name in object_size:
                size = object_size[label_name]
            if size > 1:
                scale_label = scipy.ndimage.binary_dilation(scale_label,
                                        iterations=size / 2)
            scale_label[:PATCH_SIZE / 2, :] = 0
            scale_label[-PATCH_SIZE / 2 - 1:, :] = 0
            scale_label[:, :PATCH_SIZE / 2] = 0
            scale_label[:, -PATCH_SIZE / 2 - 1:] = 0
            for y, x in np.c_[np.where(scale_label)]:
                patch = image[y - PATCH_SIZE / 2 : y + PATCH_SIZE / 2 + 1,
                              x - PATCH_SIZE / 2 : x + PATCH_SIZE / 2 + 1]
                our_patches.append(patch.ravel())
                our_patch_labels.append(label_name)
            num_label_patches += len(our_patches)

            if SURROUND_SIZE <= 1:
                continue
            surround_label = scipy.ndimage.binary_dilation(scale_label,
                                        iterations=SURROUND_SIZE / 2)
            surround_label &= ~ scale_label
            surround_label[:PATCH_SIZE / 2, :] = 0
            surround_label[-PATCH_SIZE / 2 - 1:, :] = 0
            surround_label[:, :PATCH_SIZE / 2] = 0
            surround_label[:, -PATCH_SIZE / 2 - 1:] = 0
            for y, x in np.c_[np.where(surround_label)]:
                patch = image[y - PATCH_SIZE / 2 : y + PATCH_SIZE / 2 + 1,
                              x - PATCH_SIZE / 2 : x + PATCH_SIZE / 2 + 1]
                our_patches.append(patch.ravel())
                our_patch_labels.append('background')
            num_label_patches += len(our_patches)

            patches.append(our_patches)
            patch_labels.append(our_patch_labels)

    num_patches = [len(p) for p in patches]
    # If any class has more than the median number of patches,
    # get a random sample
    max_patches = int(np.median(num_patches))
    for i in xrange(len(patches)):
        if num_patches[i] > max_patches:
            patch_ind = np.random.choice(num_patches[i], max_patches,
                                         replace=True)
            patches[i] = np.array(patches[i])[patch_ind]
            patch_labels[i] = [patch_labels[i][ind] for ind in patch_ind]
    patches = np.concatenate(patches)
    patch_labels = [label for patch in patch_labels for label in patch]
    # Find equal number of background patches
    bg_patches = []
    while len(bg_patches) < num_label_patches:
        x = np.random.randint(PATCH_SIZE / 2,
                              image.shape[1] - PATCH_SIZE / 2)
        y = np.random.randint(PATCH_SIZE / 2,
                              image.shape[0] - PATCH_SIZE / 2)
        patch = image[y - PATCH_SIZE / 2 : y + PATCH_SIZE / 2 + 1,
                      x - PATCH_SIZE / 2 : x + PATCH_SIZE / 2 + 1]
        if patch.any():
            bg_patches.append(patch.ravel())
    patches = np.concatenate([patches, bg_patches])
    patch_labels += ["background" for patch in bg_patches]
    all_patches.append(patches)
    all_patch_labels.append(patch_labels)
patches = np.concatenate(all_patches)
patch_labels = [label for patch in all_patch_labels for label in patch]
bg = np.array([label == 'background' for label in patch_labels])
num_bg = sum(bg)
# Make background patches 90% of total
max_num_bg = (len(patches) - num_bg) * 9
if num_bg > max_num_bg:
    print 'pruning', num_bg, 'background to', max_num_bg
    choice = np.random.choice(num_bg, max_num_bg, replace=False)
    bg_inds = np.where(bg)[0][choice]
    non_bg_inds = np.where(~bg)[0]
    keep = np.sort(np.concatenate([bg_inds, non_bg_inds]))
    patches = patches[keep]
    patch_labels = [l for i,l in enumerate(patch_labels) if i in keep]

# Load additional manually added patches
patch_f = open('patches.csv')
new_patches = []
for line in patch_f.readlines():
    patch, label = line.strip().split(',')
    patch = np.array(list(patch), int)
    new_patches.append(patch.astype(bool))
    patch_labels.append(label)
patches = np.concatenate([patches, new_patches])

features = (patches != 0).astype(int)
patches = features.reshape((-1, 35, 35))
proj_patch = patches[:, (35-15)/2:(35+15)/2, (35-15)/2:(35+15)/2].astype(int)
assert proj_patch.shape[1:] == (15,15)
vert_proj = proj_patch.sum(1)
horiz_proj = proj_patch.sum(2)
features = np.c_[features, vert_proj, horiz_proj]

def convert_forest(classifier):
    num_trees = len(classifier.estimators_)
    tree_size = [len(e.tree_.feature) for e in classifier.estimators_]
    forest_size = sum(tree_size)
    features = np.empty(forest_size, np.int32)
    threshold = np.empty(forest_size, np.int32)
    children = np.empty((forest_size, 2), np.int32)
    # Map each root to the first n indices which each worker starts on,
    # then concatenate all remaining nodes from each tree
    nonroot_start_ind = num_trees + \
        np.cumsum([0] + [s - 1 for s in tree_size[:-1]])
    for i, tree in enumerate(classifier.estimators_):
        features[i] = tree.tree_.feature[0]
        threshold[i] = np.ceil(tree.tree_.threshold[0]).astype(np.int32)
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
        threshold[start_ind : start_ind + tree_size[i]-1] = \
            np.ceil(tree.tree_.threshold[1:]).astype(np.int32)
        orig_children = np.c_[tree.tree_.children_left[1:],
                              tree.tree_.children_right[1:]]
        new_children = (start_ind - 1 + orig_children).astype(np.int32)
        new_children[orig_children < 0] = -1
        children[start_ind : start_ind + tree_size[i]-1, :] = new_children
    return dict(num_trees=num_trees,
                features=features,
                threshold=threshold,
                children=children,
                classes=classifier.classes_)

rf = RandomForestClassifier(n_estimators=512, max_depth=8, n_jobs=-1)
rf.fit(features, patch_labels)

cPickle.dump(convert_forest(rf), open('classifier.pkl', 'wb'))
