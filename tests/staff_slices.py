import env
import moonshine
from moonshine.staves import validation
from moonshine.structure import orientation, staffsize
import numpy as np
import gzip

import sys
import cPickle

pages = moonshine.open(sys.argv[1])
output = sys.argv[2]

staves = []
for i, page in enumerate(pages):
    staffsize.staffsize(page)
    if type(page.staff_dist) is tuple or page.staff_dist is None:
        continue
    orientation.rotate(page)
    staffsize.staffsize(page)

    validator = validation.StaffValidation(page)
    scores = validator.score_staves()
    ns = page.staves.nostaff().get()
    for i, score in enumerate(scores['score'][:-1]):
        staff = page.staves()[i]
        y0 = int(staff[:,1].min() - page.staff_dist*2)
        y1 = int(staff[:,1].max() + page.staff_dist*2)
        if score >= 0.9 and y0 >= 0:
            img = ns[y0:y1]
            img = np.unpackbits(img).reshape((-1, 4096))
            img = img[:, :page.orig_size[1]]
            s = staff.copy()
            s[:, 1] -= y0
            staves.append(dict(img=img,
                               staff_line=s,
                               staff_dist=page.staff_dist))

if len(staves):
    cPickle.dump(staves, gzip.open(output, 'w'))
