import moonshine
from moonshine import preprocessing, forest, bitimage, opencl
import gc
from forest_config import COLOR_LABELS
import sys
import os
import numpy

IMSLP = '../IMSLP'
all_imslp = list(numpy.random.choice(os.listdir(IMSLP), 10))
CORPUS = [os.path.join(IMSLP,p) for p in all_imslp]

images = []
class_num = []

for score in CORPUS:
    try:
        score = moonshine.open(score)
    except Exception:
        continue
    while len(score):
        page = score[0]
        preprocessing.process(page)
        if type(page.staff_dist) is not tuple and page.staff_dist is not None and page.staff_dist >= 8:
            image, scale = forest.scale_img(page)
            pred = forest.predict(forest.classifier, image, get_classes=False)
            images.append(bitimage.as_hostimage(image))
            class_num.append(pred.get())
            del image
        del page
        del score[0]
        opencl.q.finish()
        gc.collect()
        if len(images) > 25:
            break


import numpy as np

nums=np.concatenate([c.ravel() for c in class_num])
NUM_PATCHES = 100
CUTOFF = np.percentile(nums, 100.0*NUM_PATCHES/len(nums))

print 'cutoff', CUTOFF, 'num patches', sum([np.sum(c <= CUTOFF) for c in class_num])
i=0
def get_images():
    global i
    for img, cls in zip(images, class_num):
        y, x = np.where(cls <= CUTOFF)
        img = np.pad(img, 35/2, 'constant', constant_values=0)
        y = y + 35/2
        x = x + 35/2
        for yval, xval in zip(y, x):
            c=35/2
            patch = np.empty((35, 35, 3), np.uint8)
            img_patch = img[yval-c:yval+c+1, xval-c:xval+c+1]
            patch[:] = np.where(img_patch, 0, 255)[:,:,None]
            patch[c-5:c+6,c,0] ^= 0xFF
            patch[c,c-5:c+6,0] ^= 0xFF
            #patch[[c-1,c-1,c+1,c+1],[c-1,c+1]*2,0] ^= 0xFF
            patch[c-1:c+2, c-1:c+2, 1] ^= 0xFF
            i += 1
            yield img_patch, patch
            if i == NUM_PATCHES: return

outfile = open('unlabeled_patches', 'a')
for img, p in get_images():
    outfile.write(''.join(map(str, (img.ravel() != 0).astype(int))) + '\n')
outfile.close()
