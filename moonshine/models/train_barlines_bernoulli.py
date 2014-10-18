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
import zipfile

def fit(files):
    HMM_FILE = os.path.join(os.path.dirname(__file__), 'barlines_hmm_bernoulli10.pkl')
    STAVES_FILE = os.path.join(os.path.dirname(__file__), 'unlabeled_barlines.zip')
    NUM_STAVES = 100

    model = hmm.BernoulliHMM(n_components=10)
    staves = []
    staves_zip = zipfile.ZipFile(STAVES_FILE, 'r')
    i = 0
    for staves_name in staves_zip.namelist():
        staff_bits = np.fromstring(staves_zip.open(staves_name,'r').read(), np.uint8)
        staff_bytes = np.unpackbits(staff_bits)
        staves.append(staff_bytes.reshape((20, -1)).T.astype(bool))
        i += 1
        if i >= NUM_STAVES:
            break
    print len(staves), 'staves'

    model.fit(staves)
    cPickle.dump(model, open(HMM_FILE, 'wb'))

if __name__ == '__main__':
    fit(sys.argv[1:])
