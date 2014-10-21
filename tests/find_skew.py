import env
import moonshine
from moonshine import staffsize, orientation
import numpy as np

import gc
import re
import sys

output = open('tests/skew.csv', 'ab')

for doc in sys.argv[1:]:
    num = re.search('IMSLP([0-9]+)', doc).group(0)
    try:
        pages = moonshine.open(doc)
    except Exception:
        continue
    for i, page in enumerate(pages):
        thick, space = staffsize.staffsize(page)
        if type(space) is not int:
            continue
        patches = orientation.patch_orientation(page)
        orientations = patches[:,:,0]
        scores = patches[:,:,1]
        score_cutoff = min(1.5, np.ma.median(scores) * 2)
        good = orientations[scores >= score_cutoff]
        rotation = float(np.ma.mean(good))
        skew = float(np.ma.std(good))
        skew /= np.sqrt(np.ma.sum(scores >= score_cutoff))
        output.write(','.join([num, str(i), str(rotation), str(skew)]) + '\n')
        output.flush()
