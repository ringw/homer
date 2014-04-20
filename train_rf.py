import numpy as np
from sklearn.ensemble import RandomForestClassifier
import moonshine
from moonshine import staffsize, rotate
import scipy.misc
import glob
import os.path
import cPickle

COLOR_LABELS = {
    (255, 0, 0): "empty_note",
    (0, 0, 255): "filled_note",
    (1, 0, 0): "treble_clef",
    (2, 0, 0): "small_treble_clef",
    (3, 0, 0): "bass_clef",
    (4, 0, 0): "small_bass_clef",
    (255, 255, 0): "flat",
    (255, 0, 255): "natural",
    (0, 255, 255): "sharp"
}
PATCH_SIZE = 17
LABEL_SIZE = 3

labeled_data = glob.glob('labels/*.png')
patches = np.zeros((0, PATCH_SIZE * PATCH_SIZE), np.bool)
patch_labels = []
for labels in labeled_data:
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
        is_label = label_img[:, :, :3] == np.array([[label_type]])
        label_y, label_x = np.where(is_label.all(axis=-1))
        for x, y in zip(label_x, label_y):
            x = int(x * image_scale)
            y = int(y * image_scale)
            xmin = max(PATCH_SIZE / 2, x - LABEL_SIZE / 2)
            xmax = min(image.shape[1] - PATCH_SIZE / 2 - 1, x + LABEL_SIZE / 2)
            ymin = max(PATCH_SIZE / 2, y - LABEL_SIZE / 2)
            ymax = min(image.shape[0] - PATCH_SIZE / 2 - 1, y + LABEL_SIZE / 2)
            for x_ in xrange(xmin, xmax):
                for y_ in xrange(ymin, ymax):
                    patch = image[y_ - PATCH_SIZE / 2 : y_ + PATCH_SIZE / 2 + 1,
                                  x_ - PATCH_SIZE / 2 : x_ + PATCH_SIZE / 2 + 1]
                    our_patches.append(patch.ravel())
        patches = np.concatenate([patches, our_patches])
        patch_labels += [label_name for i in our_patches]
        num_label_patches += len(our_patches)

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

rf = RandomForestClassifier()
rf.fit(patches, patch_labels)
cPickle.dump(rf, open('classifier.pkl', 'wb'))
