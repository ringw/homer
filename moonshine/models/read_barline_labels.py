import moonshine
from moonshine import staffsize, staves
import numpy as np

import glob
import os
from PIL import Image

LABELS_DIR = 'labels/barlines'

CLASSES = 'BACKGROUND BARLINE END_BAR SECTION_BAR REPEAT_LEFT REPEAT_RIGHT'.split()
LABELS = dict(((255, 0, 0), 'BARLINE'),
              ((0, 255, 0), 'END_BAR'),
              ((0, 0, 255), 'SECTION_BAR'),
              ((255, 255, 0), 'REPEAT_LEFT'),
              ((255, 0, 255), 'REPEAT_RIGHT'))

labeled_staves = []
for label in glob.glob(LABELS_DIR + '/*.png'):
    label_img = np.array(Image.open(label).convert('RGB')).astype(np.uint8)
    is_labeled = label_img.any(axis=2)
    image_path = 'samples/' + os.path.basename(label)
    page, = moonshine.open(image_path)
    staffsize.staffsize(page)
    for staff in page.staves():
        y = np.ma.median(staff[:, 1])
        y0 = max(0, y - page.staff_dist*3)
        y1 = min(page.byteimg.shape[0], y + page.staff_dist*3)
        x0 = np.ma.min(staff[:, 0])
        x1 = np.ma.max(staff[:, 1])
        col_labels = np.zeros(x1-x0, int)
        for col in xrange(x1-x0):
            x = col + x0
            has_label, = np.where(is_labeled[:, x])
            if len(has_label):
                col_labels[col] = CLASSES.find(LABELS[tuple(label_img[has_label[0],x])])
        labeled_staves.append((page.byteimg[y0:y1, x0:x1], col_labels))
