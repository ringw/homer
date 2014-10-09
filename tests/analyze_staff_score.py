import pandas as pd
import glob
import gzip
import re

#FILES = glob.glob('tests/validated/*.csv.gz')
FILES = ['staves.csv.gz']
staves = pd.DataFrame()
pages = pd.DataFrame()
for f in FILES:
    doc = pd.DataFrame.from_csv(gzip.open(f))
    our_staves = doc[~doc.index.to_series().str.contains('page')]
    page_staves = our_staves[our_staves['score'] >= 0.5]
    staves = staves.append(our_staves)
    pagenum = page_staves.index.to_series().str.replace('S.+$','') + 'page'
    staffbypage = page_staves['runs'].groupby(pagenum)
    new_page_removed = staffbypage.sum()
    new_page_df = pd.DataFrame(dict(id=new_page_removed.index,
                                    new_removed=new_page_removed),
                               index=new_page_removed.index)
    orig_pages = doc[doc.index.to_series().str.contains('page')]
    merged = orig_pages.join(new_page_df, how='outer')
    our_pages = merged[['new_removed','runs']]
    clm = list(our_pages.columns)
    clm[0] = 'removed'
    our_pages.columns = clm
    our_pages['removed'].fillna(0, inplace=True)
    our_pages['score'] = our_pages['removed'] / our_pages['runs']
    pages = pages.append(our_pages)

pages['score'].fillna(0, inplace=True)
staves['score'].fillna(0, inplace=True)

# can't figure out how to use groupby object
page_method = pages.index.to_series().str.replace('-.+','')
page_bymethod = dict(list(pages['score'].groupby(page_method)))

staff_method = staves.index.to_series().str.replace('-.+','')
staff_bymethod = dict(list(staves['score'].groupby(staff_method)))

import pylab
def density(points, *opts, **kwopts):
    from scipy import stats
    dens = stats.kde.gaussian_kde(points)
    x = pylab.arange(0, 1.5, .01)
    pylab.plot(x, dens(x), *opts, **kwopts)
