import pandas
import numpy as np

pages = pandas.DataFrame.from_csv(open('pagedata-staves.csv'))
index = pandas.Series(list(pages.index)).str.extract('^(?P<method>[a-z]+)-(?P<doc>[a-z0-9]+)-(?P<noise>.+)')
multiindex = pandas.MultiIndex.from_tuples([tuple(index.iloc[i]) for i in xrange(len(index))])
pages.index = multiindex
pageMethods = pages.unstack(1).fillna(0)
sens = pageMethods.staff_sens.apply(np.mean, 1).unstack().fillna(0)
spec = pageMethods.staff_spec.apply(np.mean, 1).unstack().fillna(0)
