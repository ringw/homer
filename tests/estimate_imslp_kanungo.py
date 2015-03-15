import env
import gzip
import metaomr
import metaomr.kanungo
import glob
import os.path
import pandas as pd
import numpy as np
from datetime import datetime
import gc
import sys

results = pd.DataFrame()
index = []
path = sys.argv[1]
name = os.path.basename(path)
print name
for i, page in enumerate(sorted(glob.glob(path+"/img-*.pbm"))):
    page, = metaomr.open(page)
    try:
        page.preprocess()
    except Exception, e:
        print e
        results = results.append([np.repeat(np.nan, 8)])
        continue
    try:
        for method, fn in (('ks', metaomr.kanungo.test_hists_ks),
                           ('m', metaomr.kanungo.test_hists_mahalanobis)):
            tic = datetime.now()
            result = metaomr.kanungo.est_parameters(page, test_fn=metaomr.kanungo.test_hists_mahalanobis)
            toc = datetime.now()
            index.append((name, i, method))
            results = results.append([tuple(result.x) + ((toc - tic).total_seconds(), result.fun)])
        del page
        gc.collect()
    except Exception, e:
        print e
        results = results.append([np.repeat(np.nan, 8)])
        continue
results.columns = 'nu a0 a b0 b k time fun'.split()
results.index = pd.MultiIndex.from_tuples(index, names=['doc','page','fn'])
results.to_csv('results/imslp_kanungo/' + name + '.csv')
