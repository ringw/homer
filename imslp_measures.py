import metaomr
import metaomr.score_matching
import metaomr.image
import glob
import os.path
import pandas as pd
import gc
import sys

results = pd.DataFrame()
path = sys.argv[1]
name = os.path.basename(path)
print name
for i, page in enumerate(sorted(glob.glob(path+"/img-*.pbm"))):
    page, = metaomr.open(page)
    page.process()
    score_results = metaomr.score_matching.score_measure_moments(page, title=name, p=i, m=len(results))
    del page
    gc.collect()
    results = results.append(score_results)
results.to_csv(sys.argv[2])
