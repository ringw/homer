import moonshine
from moonshine import preprocessing, forest, bitimage, opencl
import gc
from forest_config import COLOR_LABELS
import sys
import os
import numpy

IMSLP = '../IMSLP'
all_imslp = list(numpy.random.choice(os.listdir(IMSLP), 100))
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
        if len(images) > 1000:
            break

from Tkinter import *
import ImageTk
from PIL import Image
import numpy as np

CUTOFF=10
print 'cutoff', CUTOFF, 'num patches', sum([np.sum(c == CUTOFF) for c in class_num])
def get_images():
    for img, cls in zip(images, class_num):
        y, x = np.where(cls == CUTOFF)
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
            yield img_patch, patch

t=Tk()

classes = ["background"] + sorted(COLOR_LABELS.values())
patches_f = open('patches.csv', 'a')
patch = None
gen = get_images()
label = Label(t)
label.grid(row=0, column=0, rowspan=len(classes))
def button(x):
    global real_img, patch, img, imgTk
    if x is not None:
        print x
        vals = map(str, (real_img != 0).ravel().astype(int))
        print >> patches_f, "".join(vals) + "," + x
    try:
        real_img, patch = gen.next()
    except StopIteration:
        sys.exit(0)
    img = Image.fromarray(patch)
    imgTk = ImageTk.PhotoImage(img.resize((35*8,35*8)))
    label.configure(image=imgTk)
button(None)
def makeCallback(class_name, narg=0):
    # weird issue with scope when trying to make a lambda in a for loop
    if narg == 0:
        return lambda: button(class_name)
    else:
        assert narg == 1
        return lambda x: button(class_name)
# assign already mapped keys
keypress = dict()
for i,c in enumerate(classes):
    for charnum,char in enumerate(c):
        if char not in keypress and char in map(chr,range(ord('a'),ord('z'))):
            keypress[char] = c
            break
    b = Button(t, text=c, width=50, command=makeCallback(c), underline=charnum)
    t.bind('<Alt-%s>' % char, makeCallback(c, narg=1))
    b.grid(column=1, row=i)

t.focus()
t.mainloop()
