import moonshine
from ..barlines import hmm as bhmm
from .. import bitimage
from ..staves import validation
from . import synthesize_barlines
import numpy as np

from hmmlearn import hmm

import cPickle
import gc
import gzip
import sys
import os
import subprocess
import shutil
import glob
import tempfile
from cStringIO import StringIO
from PIL import Image
import zipfile
import re

UNLABELED_BARLINES = 'moonshine/models/unlabeled_barlines.zip'
barlines_out = zipfile.ZipFile(UNLABELED_BARLINES, 'a', zipfile.ZIP_DEFLATED)

files = sys.argv[1:]
try:
    tmpdir = tempfile.mkdtemp()
    for filename in files:
        subprocess.call(['/usr/bin/pdfimages', filename, tmpdir + '/page'])
        pages = glob.glob(tmpdir + '/page-*.pbm')
        if not (0 < len(pages) <= 100):
            for page in pages:
                os.unlink(page)
            continue
        print filename
        imslpnum = re.search(r'IMSLP([0-9]+)', filename).group(0)
        staffnum = 0
        for page in pages:
            gc.collect()
            page = moonshine.open(page)[0]
            page.preprocess()
            if type(page.staff_dist) is not int:
                continue
            page.staves()

            validator = validation.StaffValidation(page)
            scores = validator.score_staves()

            for i in xrange(len(page.staves())):
                if scores.iloc[i]['score'] > 0.80:
                    staff = bhmm.scaled_staff(page, i)
                    bs = bitimage.as_hostimage(staff)
                    #imgstr = StringIO()
                    #img = Image.fromarray(bs.astype(np.uint8)*255).convert('1')
                    #img.save(imgstr, format='png')
                    #imgstr.seek(0)
                    arcname = imslpnum + '-%d' % staffnum + '.bin'
                    #barlines_out.writestr(arcname, imgstr.read())
                    #doc_output.write(','.join(map(str,bs.shape)) + '\0')
                    #doc_output.write(np.packbits(bs).tostring())
                    barlines_out.writestr(arcname, np.packbits(bs).tostring())
                    #doc_output.write('\0')
                    staffnum += 1
        for page in pages:
            os.unlink(page)
finally:
    shutil.rmtree(tmpdir)
