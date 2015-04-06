# Automatically generate beethoven_sonata_movements.csv
import env
import metaomr
from metaomr import bitimage, deskew

import glob
import numpy as np
import os
import itertools
import re
import sys
from PIL import Image
import gc
import pandas as pd

sonatas = pd.DataFrame.from_csv('resources/beethoven_sonatas.csv', header=None)

dir_path = sys.argv[1]
imslpid = re.search('IMSLP[0-9]+', dir_path).group(0)
if imslpid not in sonatas.index:
    sys.exit(0)
pages = sorted(glob.glob(os.path.join(dir_path, '*.pbm')))
pages = [metaomr.open(page)[0] for page in pages]
if len(pages) == 0:
    sys.exit(0)
for p, page in enumerate(pages):
    sys.stderr.write('%d ' % p)
    try:
        page.preprocess()
        if type(page.staff_dist) is not int:
            raise Exception('staffsize failed')
        page.layout()
        gc.collect()
    except Exception, e:
        print imslpid, p, '-', e
        pages[p] = None

mvmt_start = []
lastpage = None
lastp = None
for p, page in enumerate(pages):
    if page is None or type(page.staff_dist) is not int:
        continue
    lastpage, lastp = page, p
    ss = page.staves()[:, 0, 0].compressed()
    if len(ss) < 4:
        continue
    if not mvmt_start:
        # Start a new movement on this page
        mvmt_start.append((p, 0))
    else:
        med = np.median(ss)
        starts = ss > med + 50
        for s, syst in enumerate(page.systems):
            if starts[syst['start']:syst['stop']+1].all():
                mvmt_start.append((p, s))
mvmt_start.append((lastp, len(lastpage.systems)))

movements = open('results/beethoven_movements.csv', 'a')
print >> movements, ','.join(map(str, itertools.chain(*(
            [(imslpid, sonatas.ix[imslpid][1])] + mvmt_start))))
