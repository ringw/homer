from Tkinter import *
from PIL import ImageTk
from PIL import Image
import numpy as np
import os
import subprocess
import sys
import tempfile
import glob
import shutil
import json

pdfpath = sys.argv[1]
jsonpath = sys.argv[2]
tmpdir = tempfile.mkdtemp()
subprocess.call(['/usr/bin/env', 'pdfimages', pdfpath, os.path.join(tmpdir,'p')])
images = sorted(glob.glob(os.path.join(tmpdir,'p-*.pbm')))
if not images:
    sys.exit(1)

def get_images():
    for img in images:
        yield Image.open(img)

t=Tk()
t.geometry("800x1000")

pages = []
staves = None
gen = get_images()
cv = Canvas(t, width=800, height=1000)
cv.pack()
def next_image(x):
    global img, imgTk, scale, pages, staves
    try:
        if staves is not None:
            pages.append(staves)
        img = gen.next()
    except StopIteration:
        json.dump(pages, open(jsonpath, 'wb'))
        sys.exit(0)
    imgsize = img.size
    scale = min(800.0/imgsize[0], 1000.0/imgsize[1])
    newsize = map(int, [imgsize[0]*scale, imgsize[1]*scale])
    imgTk = ImageTk.PhotoImage(img.convert('L').resize(newsize, Image.ANTIALIAS))
    cv.delete('all')
    cv.create_image(newsize[0]/2,newsize[1]/2,image=imgTk)
next_image(None)
staves = []
cur_staff = None
last_line = None
def click(e):
    global staves, cur_staff, last_line
    img_point = [int(e.x / scale), int(e.y / scale)]
    if cur_staff is None:
        cur_staff = [img_point]
    else:
        x0 = int(cur_staff[-1][0] * scale)
        y0 = int(cur_staff[-1][1] * scale)
        last_line = cv.create_line(x0, y0, e.x, e.y, fill='red')
        cur_staff.append(img_point)
def remove_line(e):
    global cur_staff, last_line
    if last_line:
        cv.delete(last_line)
        last_line = None
        cur_staff = cur_staff[:-1]
def key(e):
    if e.char == ' ':
        global staves, cur_staff
        assert cur_staff
        staves.append(cur_staff)
        cur_staff = None

t.bind('<Return>', next_image)
cv.bind('<Button-1>', click)
t.bind('<BackSpace>', remove_line)
t.bind('<Key>', key)
t.focus()
t.mainloop()
shutil.rmtree(tmpdir)
