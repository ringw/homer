from Tkinter import *
import ImageTk
from forest_config import COLOR_LABELS
from PIL import Image
import numpy as np
import os
from moonshine import forest
from cPickle import load

# Read already labeled patches and skip duplicates in unlabeled_patches
already_labeled = set()
for line in open('patches.csv').readlines():
    patch = line.split(',')[0]
    already_labeled.add(patch)

def get_images():
    # Read in images from unlabeled_patches
    c=35/2
    for line in open('unlabeled_patches').readlines():
        patch_str = line.strip()
        if patch_str in already_labeled:
            continue
        img_patch = np.array(list(patch_str),int).reshape((35,35))
        patch = np.empty((35, 35, 3), np.uint8)
        patch[:] = np.where(img_patch, 0, 255)[:,:,None]
        patch[c-5:c+6,c,0] ^= 0xFF
        patch[c,c-5:c+6,0] ^= 0xFF
        patch[c-1:c+2, c-1:c+2, 1] ^= 0xFF
        yield img_patch, patch
    os.unlink('unlabeled_patches')

t=Tk()

classes = ["background"] + sorted(COLOR_LABELS.values()) + ['skip','quit']
patches_f = open('patches.csv', 'a')
patch = None
gen = get_images()
label = Label(t)
label.grid(row=0, column=0, rowspan=len(classes))
buttons = dict()
def button(x):
    global real_img, patch, img, imgTk
    if x == 'skip':
        pass
    elif x == 'quit':
        sys.exit(0)
    elif x is not None:
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
    pred_cls = np.array(classes)[forest.predict_cpu(forest.cpu_classifier, real_img, 2)]
    for c,b in buttons.iteritems():
        if c == pred_cls[0]:
            b.configure(background='orange')
        elif c == pred_cls[1]:
            b.configure(background='yellow')
        else:
            b.configure(background='white')
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
    buttons[c] = b
button(None)

t.focus()
t.mainloop()
