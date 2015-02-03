import metaomr
from metaomr import preprocessing, forest, bitimage
import scipy.misc
import sys
from timeit import Timer
import numpy as np
#from pylab import *

image, = metaomr.open(sys.argv[1])
preprocessing.process(image)
assert type(image.staff_dist) is not tuple and image.staff_dist >= 8
image_scale = 8.0 / image.staff_dist
image = scipy.misc.imresize(image.byteimg[:image.orig_size[0],
                                          :image.orig_size[1]].astype(bool),
                            image_scale,
                            interp='nearest')
padded_image = np.zeros((2048, 2048), np.uint8)
padded_image[:image.shape[0], :image.shape[1]] = image
bit_image = bitimage.as_bitimage(padded_image)
t = Timer(lambda: forest.predict(forest.classifier, bit_image))
print t.timeit(number=1)
#display_image = zeros(image.shape + (3,), int8)
#display_image[:] = np.where(image, 0, 255)[:, :, None]
#PATCH_SIZE = 17
#for y in xrange(PATCH_SIZE / 2, image.shape[0] - PATCH_SIZE/2 - 1):
#    patches = np.zeros((image.shape[1] - (PATCH_SIZE/2)*2 - 1, PATCH_SIZE*PATCH_SIZE), int)
#    for i,x in enumerate(xrange(PATCH_SIZE / 2, image.shape[1] - PATCH_SIZE/2 - 1)):
#        patches[i] = image[y - PATCH_SIZE/2 : y + PATCH_SIZE/2 + 1,
#                      x - PATCH_SIZE/2 : x + PATCH_SIZE/2 + 1].ravel()
#    pred_class = classifier.predict(patches)
#    for i,x in enumerate(xrange(PATCH_SIZE / 2, image.shape[1] - PATCH_SIZE/2 - 1)):
#        if pred_class[i] == "filled_note":
#            display_image[y, x, 2] ^= 255
#        elif pred_class[i] == "empty_note":
#            display_image[y, x, 0] ^= 255
#        elif pred_class[i] == "bass_clef":
#            display_image[y, x, 1] ^= 255
#
#imshow(display_image)
#show()
