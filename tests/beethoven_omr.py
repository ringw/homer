import env
import metaomr

import glob
import numpy as np
import os.path
import pandas as pd
import re
import sys
from PIL import Image

sonatas = pd.DataFrame.from_csv('resources/beethoven_sonata_movements.csv',
                                header=None)

dir_path = sys.argv[1]
output_path = sys.argv[2]
name = os.path.basename(dir_path)
imslpid = re.search('IMSLP[0-9]+', name).group(0)
pages = sorted(glob.glob(os.path.join(dir_path, '*.pbm')))

info = sonatas.ix[imslpid]
mvmt_start = np.array(info[~ info.isnull()][1:]).reshape((-1, 2))

pages = [metaomr.open(page)[0] for page in pages]
for page in pages:
    page.process()

for movement in xrange(len(mvmt_start) - 1):
    mvmt_path = os.path.join(output_path, 'mvmt%d' % movement)
    os.path.mkdir(mvmt_path)
    for p in xrange(mvmt_start[movement, 0],
                    mvmt_start[movement+1, 0]
                    if mvmt_start[movement+1, 1] == 0
                    else mvmt_start[movement+1, 0] + 1):
        page = pages[p]
        if page == mvmt_start[movement, 0] and mvmt_start[movement, 1] > 0:
            b = page.boundaries[page.systems[mvmt_start[movement,1]]['stop']+1]
            boundary = np.repeat(b[:,1], b[1,0] - b[0,0])
            img = page.byteimg[:, :min(page.orig_size[1], len(boundary))]
            boundary = boundary[:img.shape[1]]
            img[np.arange(img.shape[0])[:, None] > boundary[None, :]] = 0
        elif (page == mvmt_start[movement+1, 0]
              and 0 < mvmt_start[movement+1, 1] < len(page.systems)):
            b = page.boundaries[page.systems[mvmt_start[movement,1]]['stop']+1]
            boundary = np.repeat(b[:,1], b[1,0] - b[0,0])
            img = page.byteimg[:, :min(page.orig_size[1], len(boundary))]
            boundary = boundary[:img.shape[1]]
            img[np.arange(img.shape[0])[:, None] <= boundary[None, :]] = 0
        else:
            img = page.byteimg[:, :page.orig_size[1]]
        img = Image.fromarray(np.where(img, 0, 255).astype(np.uint8))
        img.save(os.path.join(mvmt_path, 'page%d.tif' % p))
