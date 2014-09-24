import pandas as pd
import glob
import gzip
import re

FILES = glob.glob('tests/validated/*.csv.gz')
staves = pd.DataFrame()
pages = pd.DataFrame()
for f in FILES:
    docname = re.search('IMSLP([0-9]+)', f).group(0)
    doc = pd.DataFrame.from_csv(gzip.open(f))
    our_staves = doc[~doc['id'].str.contains('page')]
    our_staves = our_staves[our_staves['score'] >= 0.25]
    our_staves['doc'] = docname
    staves = staves.append(our_staves)
    pagenum = our_staves['id'].str.replace('S.+$','') + 'page'
    staffbypage = our_staves['runs'].groupby(pagenum)
    new_page_removed = staffbypage.sum()
    new_page_df = pd.DataFrame(dict(id=new_page_removed.index,
                                    new_removed=new_page_removed))
    orig_pages = doc[doc['id'].str.contains('page')]
    merged = pd.merge(orig_pages, new_page_df, how='outer', on='id')
    our_pages = merged[['id','new_removed','runs']]
    clm = list(our_pages.columns)
    clm[1] = 'removed'
    our_pages.columns = clm
    our_pages['removed'].fillna(0, inplace=True)
    our_pages['score'] = our_pages['removed'] / our_pages['runs']
    our_pages['doc'] = docname
    pages = pages.append(our_pages)

# can't figure out how to use groupby object
page_method = pages['id'].str.replace('P.+','')
page_bymethod = dict(list(pages['score'].groupby(page_method)))

staff_method = staves['id'].str.replace('P.+','')
staff_bymethod = dict(list(staves['score'].groupby(staff_method)))

import pylab
def density(points, *opts, **kwopts):
    from scipy import stats
    dens = stats.kde.gaussian_kde(points)
    x = pylab.arange(0, 1.5, .01)
    pylab.plot(x, dens(x), *opts, **kwopts)
