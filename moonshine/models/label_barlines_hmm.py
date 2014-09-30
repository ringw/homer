import moonshine
from ..barlines import hmm as bhmm
from .. import bitimage
from ..staves import validation

from hmmlearn import hmm

import cPickle
import gc
import numpy as np
import sys
import os

def label(files):
    HMM_FILE = os.path.join(os.path.dirname(__file__), 'barlines_hmm.pkl')
    HMM_LABELS_FILE = os.path.join(os.path.dirname(__file__), 'barlines_hmm_labels.txt')
    labels_out = open(HMM_LABELS_FILE, 'wb')
    model = cPickle.load(open(HMM_FILE))

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
    allstates = []
    for staff in staves:
        logp, states = model.decode(staff)
        allstates.append(states)
    allstates = np.concatenate(allstates)
    repstaves = [staff for staff in staves for i in xrange(len(staff))]
    column_x = [i for staff in staves for i in xrange(len(staff))]
    labels = []
    for state in xrange(model.n_components):
        ours, = np.where(allstates == state)
        np.random.shuffle(ours)
        mostlikely = ours
        import pylab
        for i in xrange(min(16, len(mostlikely))):
            pylab.subplot(4, 4, i)
            ind = mostlikely[i]
            staff = repstaves[ind]
            staffmin = max(0, column_x[ind] - 20)
            staffmax = min(len(staff), column_x[ind] + 20)
            img = np.zeros((staff.shape[1], staffmax-staffmin, 3),bool)
            img[:,:,1] = staff[staffmin:staffmax].T
            img[:,column_x[ind] - staffmin,0] = 1
            pylab.imshow(img)
        pylab.show()
        labels_out.write(sys.stdin.readline().strip() + '\n')
        labels_out.flush()
        pylab.clf()

    cPickle.dump(labels, open(HMM_LABELS_FILE, 'wb'))

if __name__ == '__main__':
    label(sys.argv[1:])
