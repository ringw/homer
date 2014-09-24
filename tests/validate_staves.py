# Comparative staff detection and removal accuracy
import env
import moonshine
from moonshine.staves import validation, hough, path
from moonshine.staves.gamera_musicstaves import *
from moonshine.structure import orientation, staffsize

methods = dict(hough=hough.FilteredHoughStaves,
               path=path.StablePathStaves,
               linetracking=MusicStaves_linetracking,
               carter=MusicStaves_rl_carter,
               fujinaga=MusicStaves_rl_fujinaga,
               roach_tatem=MusicStaves_rl_roach_tatem,
               gamera_simple=MusicStaves_rl_simple,
               skeleton=MusicStaves_skeleton,
               dalitz=StaffFinder_dalitz,
               miyao=StaffFinder_miyao,
               projections=StaffFinder_projections)

import sys

pages = moonshine.open(sys.argv[1])
output = sys.argv[2] if len(sys.argv) > 2 else None

import gzip
import pandas
result = pandas.DataFrame()
for i, page in enumerate(pages):
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
            scores['id'] = ('%sP%02d' % (method, i)) + scores['id']
            result = result.append(scores)

if output is not None and len(result):
    result.to_csv(gzip.open(output, 'wb'))
