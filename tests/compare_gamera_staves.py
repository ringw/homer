import env
import os
import sys
import subprocess
import tempfile
import glob
import json
import re
import moonshine
import numpy as np
from datetime import datetime
from moonshine import structure
from moonshine.staves import hough,path

pdfpath = sys.argv[1]
gamera = sys.argv[2]
output = sys.argv[3]

tmpdir = tempfile.mkdtemp()
#subprocess.call(['/usr/bin/pdfimages', pdfpath, tmpdir + '/p'])
#pages = sorted(glob.glob(tmpdir + '/p-*.pbm'))
pages = [pdfpath]
all_staves = json.load(open(gamera))
assert len(pages) == len(all_staves)
#print re.search('IMSLP([0-9]+)',pdfpath).group(0), len(pages), 'pages'

def staff_dict(staff):
    global page
    ys=map(int, np.arange(-2,3)*mpage.staff_dist + np.mean([staff[0][1], staff[-1][1]]))
    rect = [[staff[0][0], ys[0]], [staff[1][0], ys[-1]]]
    return dict(ys=ys, rect=rect)
for i, page in enumerate(pages):
    mpage, = moonshine.open(page)
    structure.staffsize.staffsize(mpage)
    structure.orientation.rotate(mpage)
    structure.staffsize.staffsize(mpage)
    start = datetime.now()
    h = hough.FilteredHoughStaves(mpage).get_staves().tolist()
    end = datetime.now()
    all_staves[i]['hough'] = map(staff_dict, h)
    all_staves[i]['time']['hough'] = int((end-start).total_seconds()*1000)

    start = datetime.now()
    p = path.StablePathStaves(mpage).get_staves().tolist()
    end = datetime.now()
    all_staves[i]['path'] = map(staff_dict, p)
    all_staves[i]['time']['path'] = int((end-start).total_seconds()*1000)
    all_staves[i]['m_orientation'] = mpage.orientation
    all_staves[i]['m_size'] = mpage.orig_size
json.dump(all_staves, open(output, 'wb'))
