import numpy as np

from hmmlearn import hmm

import cPickle
import gc
import sys
import os
import zipfile

def fit(files):
    HMM_FILE = os.path.join(os.path.dirname(__file__), 'barlines_hmm_bernoulli100.pkl')
    STAVES_FILE = os.path.join(os.path.dirname(__file__), 'unlabeled_barlines.zip')
    NUM_STAVES = 5000

    model = hmm.BernoulliHMM(n_components=100)
    staves = []
    staves_zip = zipfile.ZipFile(STAVES_FILE, 'r')
    i = 0
    for staves_name in staves_zip.namelist():
        staff_bits = np.fromstring(staves_zip.open(staves_name,'r').read(), np.uint8)
        staff_bytes = np.unpackbits(staff_bits)
        staves.append(staff_bytes.reshape((25, -1)).T.astype(bool))
        i += 1
        if i >= NUM_STAVES:
            break
    print len(staves), 'staves'

    model._init(staves[:1])
    for s in xrange(3):
        w0 = 3*s
        b = 3*s + 1
        w1 = 3*s + 2
        emission = 0.1 * np.random.random((3, model.n_outputs))
        emission[1] += 0.9
        model._log_emissionprob[w0:w1+1] = np.log(emission)
        tr = model.transmat_[w0:w1+1]
        tr[[0,1],[1,2]] += 0.5
        tr[[1,2],[0,1]] = 1e-5
        tr /= tr.sum(1)[:, None]
        model._log_transmat[w0:w1+1] = np.log(tr)
    model.fit(staves)
    cPickle.dump(model, open(HMM_FILE, 'wb'))

if __name__ == '__main__':
    fit(sys.argv[1:])
