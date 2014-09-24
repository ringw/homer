import env
import moonshine
from moonshine.staves import validation
from moonshine.structure import orientation, staffsize

import sys

pages = moonshine.open(sys.argv[1])
output = sys.argv[2] if len(sys.argv) > 2 else None

import pandas
result = pandas.DataFrame()
for i, page in enumerate(pages):
    staffsize.staffsize(page)
    if type(page.staff_dist) is tuple or page.staff_dist is None:
        continue
    orientation.rotate(page)
    staffsize.staffsize(page)

    validator = validation.StaffValidation(page)
    scores = validator.score_staves()
    if output is None:
        print scores
    else:
        scores['id'] = ('P%02d' % i) + scores['id']
        result = result.append(scores)

if output is not None and len(result):
    result.to_csv(output)
