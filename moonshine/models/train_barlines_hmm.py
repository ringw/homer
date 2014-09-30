import moonshine
from ..barlines import hmm as bhmm
from .. import bitimage
from ..staves import validation

from hmmlearn import hmm

import cPickle
import gc
import sys
import os

def fit(files):
    HMM_FILE = os.path.join(os.path.dirname(__file__), 'barlines_hmm.pkl')

    if os.path.exists(HMM_FILE):
        model = cPickle.load(HMM_FILE)
    else:
        model = hmm.BernoulliHMM(n_components=50)

    staves = []
    for filename in files:
        try:
            pages = moonshine.open(filename)
        except Exception:
            continue
        if not (0 < len(pages) <= 50):
            continue
        for page in pages:
            gc.collect()
            page.preprocess()
            if type(page.staff_dist) is not int:
                continue
            page.staves()

            validator = validation.StaffValidation(page)
            scores = validator.score_staves()

            for i in xrange(len(page.staves())):
                if scores.loc[i]['score'] > 0.90:
                    staff = bhmm.scaled_staff(page, i)
                    staves.append(bitimage.as_hostimage(staff).T.astype(bool))
    print len(staves), "staves"
    model.fit(staves)
    cPickle.dump(model, open(HMM_FILE, 'wb'))

if __name__ == '__main__':
    fit(sys.argv[1:])
