# Comparative staff detection and removal accuracy
import env
import moonshine
from moonshine.staves import validation, hough, path
from moonshine.staves.gamera_musicstaves import *
from moonshine import orientation, staffsize

methods = dict(hough=hough.FilteredHoughStaves,
               #path=path.StablePathStaves,
               #linetracking=MusicStaves_linetracking,
               #carter=MusicStaves_rl_carter,
               fujinaga=MusicStaves_rl_fujinaga,
               #roach_tatem=MusicStaves_rl_roach_tatem,
               #gamera_simple=MusicStaves_rl_simple,
               skeleton=MusicStaves_skeleton,
               dalitz=StaffFinder_dalitz,
               #miyao=StaffFinder_miyao,
               projections=StaffFinder_projections)

import glob
import gzip
import pandas
import re
import shutil
import sys
import subprocess
import tempfile
tmpdir = tempfile.mkdtemp()

try:
    subprocess.check_output(['/usr/bin/pdfimages', sys.argv[1], tmpdir + '/orig'])
    if not (0 < len(glob.glob(tmpdir + '/orig-*.pbm')) <= 50):
        shutil.rmtree(tmpdir)
        sys.exit(0)

    output = sys.argv[2] if len(sys.argv) > 2 else None

    result = pandas.DataFrame()
    for i, pagename in enumerate(sorted(glob.glob(tmpdir + '/orig-*.pbm'))):
        page, = moonshine.open(pagename)

        staffsize.staffsize(page)
        if type(page.staff_dist) is tuple or page.staff_dist is None:
            continue
        orientation.rotate(page)
        staffsize.staffsize(page)

        validator = validation.StaffValidation(page)
        for method in methods:
            staves = methods[method](page)
            scores = validator.score_staves(method=staves)
            if output is None:
                print scores
            else:
                scores.index = ('%sP%02d' % (method, i)) + scores.index
                result = result.append(scores)
            if method in 'fujinaga skeleton'.split():
                staves = methods[method](page, staff_removal='gamera')
                scores = validator.score_staves(method=staves)
                if output is None:
                    print scores
                else:
                    scores.index = ('%s-nativeP%02d' % (method, i)) + scores.index
                    result = result.append(scores)

    if output is not None and len(result):
        result.to_csv(gzip.open(output, 'wb'))
finally:
    shutil.rmtree(tmpdir)
