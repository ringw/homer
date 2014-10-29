import pandas
import numpy as np
import gzip

labeled = pandas.DataFrame.from_csv(open('pagedata-staves.csv'))
index = pandas.Series(list(labeled.index)).str.extract('^(?P<method>[a-z]+)-(?P<doc>[a-z0-9]+)-(?P<noise>.+)')
multiindex = pandas.MultiIndex.from_tuples([tuple(index.iloc[i]) for i in xrange(len(index))])
labeled.index = multiindex
pageMethods = labeled.unstack(1).fillna(0)
sens = pageMethods.staff_sens.apply(np.mean, 1).unstack().fillna(0)
spec = pageMethods.staff_spec.apply(np.mean, 1).unstack().fillna(0)

unlabeled = pandas.DataFrame.from_csv(gzip.open('staves.csv.gz'))
ul_index = pandas.Series(list(unlabeled.index)).str.extract(
    '^(?P<method>[a-z]+(?:-native)?)-(?P<doc>[a-z0-9]+)-(?P<noise>.+?)-(?P<staffpage>S[0-9]+|page)$')
ul_multiindex = pandas.MultiIndex.from_tuples([tuple(ul_index.iloc[i]) for i in xrange(len(ul_index))])
unlabeled.index = ul_multiindex
