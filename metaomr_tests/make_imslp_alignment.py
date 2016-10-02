import env
import metaomr
import metaomr.score_matching as sm

import glob
import os.path
import re
import sys
import pandas as pd
import gzip
import cPickle
import gc

DOCS = sorted(glob.glob('imslp/measures/*.csv'))
alignments = {}
for i in xrange(len(DOCS)):
    path1 = DOCS[i]
    name1 = re.search('IMSLP[0-9]+', path1).group(0)
    doc1 = pd.DataFrame.from_csv(path1, index_col=range(5))
    for j in xrange(i+1, len(DOCS)):
        path2 = DOCS[j]
        name2 = re.search('IMSLP[0-9]+', path2).group(0)
        doc2 = pd.DataFrame.from_csv(path2, index_col=range(5))
        if abs(len(doc1) - len(doc2)) > max(len(doc1), len(doc2)) * 0.1:
            continue
        alignment = sm.align_docs(doc1, doc2)
        if alignment is not None:
            alignment.to_csv(gzip.open('imslp/alignments/%s-%s.csv.gz' % (name1,name2), 'w'))
        sys.stderr.write('.')
    sys.stderr.write('\n')
    gc.collect()
