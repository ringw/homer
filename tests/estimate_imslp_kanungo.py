import env
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
for path in sorted(glob.glob("imslp/pbm/IMSLP*")):
    name = os.path.basename(path)
    print name
    for i, page in enumerate(sorted(glob.glob(path+"/img-*.pbm"))):
        index.append((name, i))
        page, = metaomr.open(page)
        try:
            page.preprocess()
        except Exception, e:
            print e
            results = results.append([np.repeat(np.nan, 8)])
            continue
        try:
            tic = datetime.now()
            result = metaomr.kanungo.est_parameters(page)
            toc = datetime.now()
            results = results.append([tuple(result.x) + ((toc - tic).total_seconds(), result.fun)])
            del page
            gc.collect()
        except Exception, e:
            print e
            results = results.append([np.repeat(np.nan, 8)])
            continue
    results_ = results.copy()
    results_.columns = 'nu a0 a b0 b k time fun'.split()
    results_.index = pd.MultiIndex.from_tuples(index, names=['doc','page'])
    results_.to_csv('results/imslp_kanungo.csv')
