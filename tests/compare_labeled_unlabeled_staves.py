import pandas
import numpy as np
import gzip

labeled = pandas.DataFrame.from_csv(open('pagedata-staves.csv'))
index = pandas.Series(list(labeled.index)).str.extract('^(?P<method>[a-z]+)-(?P<doc>[a-z0-9]+)-(?P<noise>.+)')
multiindex = pandas.MultiIndex.from_tuples([tuple(index.iloc[i]) for i in xrange(len(index))])
labeled.index = multiindex
pageMethods = labeled.fillna(0)
pageMethods.index.names = 'method doc noise'.split()

unlabeled = pandas.DataFrame.from_csv(gzip.open('staves.csv.gz'))
ul_index = pandas.Series(list(unlabeled.index)).str.extract(
    '^(?P<method>[a-z]+(?:-native)?)-(?P<doc>[a-z0-9]+)-(?P<noise>.+?)-(?P<staffpage>S[0-9]+|page)$')
ul_multiindex = pandas.MultiIndex.from_tuples([tuple(ul_index.iloc[i]) for i in xrange(len(ul_index))], names=ul_index.columns)
unlabeled.index = ul_multiindex
unlabeled = unlabeled.iloc[~np.array(unlabeled.index.get_level_values('method').to_series().str.contains('-native'))]
# Remove duplicates
unlabeled = unlabeled.groupby(unlabeled.index).first()
unlabeled.index = pandas.MultiIndex.from_tuples(unlabeled.index, names=ul_index.columns)
ulp = unlabeled.xs('page', level='staffpage').score.fillna(0)

sens = pandas.DataFrame([pageMethods.staff_sens, ulp]).T
sens.columns = ['labeled', 'unlabeled']
