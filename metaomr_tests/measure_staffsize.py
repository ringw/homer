import env
import metaomr
from metaomr import staffsize

import gc
import gzip
import numpy as np
import os
import re
import sys

out = ''
for doc in sys.argv[1:]:
    docname = re.search('IMSLP([0-9]+)', doc).group(0)
    pages = metaomr.open(doc)
    for i, page in enumerate(pages):
        name = docname + '-' + str(i)
        dark = staffsize.dark_runs(page.img)
        staff_thick = np.argmax(dark)
        page.staff_thick = staff_thick
        light = staffsize.light_runs(page)
        out += ','.join([name] + map(str,dark) + map(str,light)) + '\n'
        gc.collect()

output = gzip.open(os.path.join('staffsizes', str(os.getpid())), 'ab')
output.write(out)
