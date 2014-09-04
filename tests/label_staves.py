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
images = glob.glob(os.path.join(tmpdir,'p-*.pbm'))
if not images:
    sys.exit(1)

def get_images():
    for img in images:
        yield Image.open(img)
    shutil.rmtree(tmpdir)

t=Tk()
t.geometry("600x800")

pages = []
staves = None
gen = get_images()
cv = Canvas(t, width=600, height=800)
cv.pack()
def next_image(x):
    global img, imgTk, scale, staves
    if staves is not None:
        pages.append(staves)
        staves = []
    try:
        img = gen.next()
    except StopIteration:
        json.dump(pages, open(jsonpath, 'wb'))
        sys.exit(0)
    imgsize = img.size
    scale = min(600.0/imgsize[0], 800.0/imgsize[1])
    newsize = map(int, [imgsize[0]*scale, imgsize[1]*scale])
    imgTk = ImageTk.PhotoImage(img.convert('L').resize(newsize, Image.ANTIALIAS))
    cv.delete('all')
    cv.create_image(newsize[0]/2,newsize[1]/2,image=imgTk)
next_image(None)
staves = []
cur_staff = None
last_line = None
staffspace_line = None
staffspace = None
def get_staffspace(x,y):
    for (x0,y0),(x1,y1) in zip(cur_staff[:-1],cur_staff[1:]):
        if x0 <= x and x < x1:
            staff_y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
            return y - staff_y
    return 0
def click(e):
    global staves, cur_staff, last_line, staffspace_line
    img_point = [int(e.x / scale), int(e.y / scale)]
    if cur_staff is None:
        cur_staff = [img_point]
    elif staffspace_line:
        staves.append(dict(center=cur_staff, staffspace=staffspace))
        cur_staff = None
        staffspace_line = None
    else:
        x0 = int(cur_staff[-1][0] * scale)
        y0 = int(cur_staff[-1][1] * scale)
        last_line = cv.create_line(x0, y0, e.x, e.y, fill='red')
        cur_staff.append(img_point)
def motion(e):
    global cur_staff, staffspace_line, staffspace
    x = e.x / scale
    y = e.y / scale
    if staffspace_line is not None:
        staffspace = get_staffspace(x, y)
        for (x0,y0), (x1,y1), spaceseg in \
                zip(cur_staff[:-1], cur_staff[1:], staffspace_line):
            cv.coords(spaceseg, x0*scale, (y0+staffspace)*scale,
                                x1*scale, (y1+staffspace)*scale)
def remove_line(e):
    global cur_staff, last_line
    if last_line:
        cv.delete(last_line)
        last_line = None
        cur_staff = cur_staff[:-1]
def key(e):
    if e.char == ' ':
        global staves, cur_staff, staffspace_line
        assert cur_staff
        if staffspace_line is None:
            staffspace_line = [cv.create_line(0,0,0,0,fill='blue')
                               for seg in cur_staff[:-1]]

t.bind('<Return>', next_image)
cv.bind('<Button-1>', click)
t.bind('<BackSpace>', remove_line)
t.bind('<Key>', key)
t.bind('<Motion>', motion)
t.focus()
t.mainloop()
