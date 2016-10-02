import env
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
    try:
        page.layout()
    except Exception, e:
        print e
        continue
    try:
        score_results = metaomr.score_matching.measure_projections(page, title=name, p=i, m=len(results))
        del page
        gc.collect()
        results = results.append(score_results)
    except Exception, e:
        print 'During measure_projections:', e
        continue
results.to_csv(sys.argv[2])
