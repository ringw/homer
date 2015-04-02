import env
import metaomr
from metaomr import bitimage, deskew

import glob
import numpy as np
import os
import pandas as pd
import re
import sys
from PIL import Image
import zipfile
from cStringIO import StringIO
import gc

sonatas = pd.DataFrame.from_csv('resources/beethoven_sonata_movements.csv',
                                header=None)

dir_path = sys.argv[1]
name = os.path.basename(dir_path)
imslpid = re.search('IMSLP[0-9]+', name).group(0)
pages = sorted(glob.glob(os.path.join(dir_path, '*.pbm')))

if imslpid not in sonatas.index:
    sys.exit()

output = zipfile.ZipFile(sys.argv[2], 'w', zipfile.ZIP_DEFLATED)
info = sonatas.ix[imslpid]
mvmt_start = np.array(info[~ info.isnull()][1:]).reshape((-1, 2))

pages = [metaomr.open(page)[0] for page in pages]
if len(pages) == 0:
    sys.exit(0)
for p, page in enumerate(pages):
    try:
        page.preprocess()
        if type(page.staff_dist) is not int:
            raise Exception('staffsize failed')
        deskew.deskew(page)
        page.process()
    except Exception, e:
        print imslpid, p, '-', e
        pages[p] = None
for movement in xrange(len(mvmt_start) - 1):
    mvmt_path = 'mvmt%d' % movement
    for p in xrange(mvmt_start[movement, 0],
                    mvmt_start[movement+1, 0]
                    if mvmt_start[movement+1, 1] == 0
                    else mvmt_start[movement+1, 0] + 1):
        page = pages[p]
        if page is None:
            continue
        img = bitimage.as_hostimage(page.img)[:, :page.orig_size[1]]
        if p == mvmt_start[movement, 0] and mvmt_start[movement, 1] > 0:
            new_system = page.systems[mvmt_start[movement,1]]['start']
            b = page.boundaries[new_system]
            boundary = np.repeat(b[:,1], b[1,0] - b[0,0])
            img = img[:, :min(page.orig_size[1], len(boundary))]
            boundary = boundary[:img.shape[1]]
            img[np.arange(img.shape[0])[:, None] < boundary[None, :]] = 0
        elif (p == mvmt_start[movement+1, 0]
              and 0 < mvmt_start[movement+1, 1] < len(page.systems)):
            new_system = page.systems[mvmt_start[movement+1,1]]['start']
            b = page.boundaries[new_system]
            boundary = np.repeat(b[:,1], b[1,0] - b[0,0])
            img = img[:, :min(page.orig_size[1], len(boundary))]
            boundary = boundary[:img.shape[1]]
            img[np.arange(img.shape[0])[:, None] >= boundary[None, :]] = 0
        img = Image.fromarray(np.where(img, 0, 255).astype(np.uint8))
        buf = StringIO()
        img.save(buf, 'tiff')
        buf.seek(0)
        output.writestr(os.path.join(mvmt_path, 'page%03d.tif' % p), buf.read())
