import cPickle
import numpy as np

def load_forest(path):
    return cPickle.load(path)

PATCH_SIZE = 17
def predict(forest, bitimg):
    image = np.unpackbits(bitimg.get()).reshape((bitimg.shape[0], -1))
    prediction = np.zeros_like(image, dtype=int)
    class_list = list(forest.classes_)
    for y in xrange(image.shape[0]):
        for x in xrange(image.shape[1]):
            image_ymin = max(0, y - PATCH_SIZE/2)
            image_ymax = min(image.shape[0], y + PATCH_SIZE/2 + 1)
            image_xmin = max(0, x - PATCH_SIZE/2)
            image_xmax = min(image.shape[1], x + PATCH_SIZE/2 + 1)
            patch = np.zeros((PATCH_SIZE, PATCH_SIZE), int)
            patch_ymin = image_ymin + PATCH_SIZE/2 - y
            patch_ymax = image_ymax + PATCH_SIZE/2 - y
            patch_xmin = image_xmin + PATCH_SIZE/2 - x
            patch_xmax = image_xmax + PATCH_SIZE/2 - x
            patch[patch_ymin:patch_ymax, patch_xmin:patch_xmax] = \
                image[image_ymin:image_ymax, image_xmin:image_xmax].copy() * 255
            #import pylab
            #pylab.subplot(16,8,x)
            #pylab.imshow(patch)
            prediction[y, x] = class_list.index(forest.predict(patch.ravel())[0])
    return prediction
