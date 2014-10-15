import moonshine
from ..barlines import hmm as bhmm
from .. import bitimage
from ..staves import validation
from . import synthesize_barlines
import numpy as np

from hmmlearn import hmm

import cPickle
import gc
import sys
import os

def fit(files):
    HMM_FILE = os.path.join(os.path.dirname(__file__), 'barlines_hmm.pkl')

    model = hmm.BernoulliMultiHMM(n_components=100)

    staves = []
    for filename in files:
        try:
            pages = moonshine.open(filename)
        except Exception:
            import traceback
            traceback.print_exc()
            continue
        if not (0 < len(pages) <= 100):
            continue
        print filename
        for page in pages:
            gc.collect()
            page.preprocess()
            if type(page.staff_dist) is not int:
                continue
            page.staves()

            validator = validation.StaffValidation(page)
            scores = validator.score_staves()

            for i in xrange(len(page.staves())):
                if scores.iloc[i]['score'] > 0.80:
                    staff = bhmm.scaled_staff(page, i)
                    staffimg = bitimage.as_hostimage(staff).T.astype(bool)
                    staves.append((staffimg, -np.ones(staffimg.shape[0],int)))
    print len(staves), "unlabeled staves"
    LABELS = 'BACKGROUND BAR THICK_BAR BAR_DOTS'.split()
    for i in xrange(len(staves)/10):
        st, lb = synthesize_barlines.gen_staff()
        labels = np.array([LABELS.index(l) for l in lb])
        staves.append((st.T, labels))
    model.fit(staves)
    cPickle.dump(model, open(HMM_FILE, 'wb'))

if __name__ == '__main__':
    fit(sys.argv[1:])
