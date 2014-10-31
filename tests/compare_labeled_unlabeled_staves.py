import pandas
import numpy as np
import gzip

labeled = pandas.DataFrame.from_csv(open('pagedata-newstaves'))
index = pandas.Series(list(labeled.index)).str.extract('^(?P<method>[a-z]+)-(?P<doc>[a-z0-9]+)-(?P<noise>.+)')
multiindex = pandas.MultiIndex.from_tuples([tuple(index.iloc[i]) for i in xrange(len(index))])
labeled.index = multiindex
pageMethods = labeled.fillna(0)
pageMethods.index.names = 'method doc noise'.split()

unlabeled = pandas.DataFrame.from_csv(open('newstaves'))
ul_index = pandas.Series(list(unlabeled.index)).str.extract(
    '^(?P<method>[a-z]+)-?(?P<native>native)?-(?P<doc>[a-z0-9]+)-(?P<noise>.+?)-?(?P<staffpage>S[0-9]+|page)$')
ul_index.native.fillna('', inplace=True)
ul_multiindex = pandas.MultiIndex.from_tuples(ul_index.apply(tuple, axis=1), names=ul_index.columns)
unlabeled.index = ul_multiindex

# Remove noise conditions which remove too many of the runs
dummy = unlabeled.xs('dummy')
runsmean = dummy.runs.groupby(map(dummy.index.get_level_values,
                                  ['noise', 'doc'])).mean()
runsremoved = runsmean.unstack(0) / runsmean.xs('orig') # XXX: broken

# XXX: need to remove 'native' in name and remove non-native counterpart
ul_methods = unlabeled.index.get_level_values('method').to_series()
ul_dummy = unlabeled.xs('dummy')
to_analyze = ~np.array(ul_methods.str.contains('-native'))
unlabeled = unlabeled.iloc[to_analyze]

# Remove duplicates
unlabeled = unlabeled.groupby(unlabeled.index).first()
unlabeled.index = pandas.MultiIndex.from_tuples(unlabeled.index, names=ul_index.columns)

allstaves = unlabeled[unlabeled.index.get_level_values('staffpage') != 'page']
allstaves.index = allstaves.index.droplevel('staffpage')
allstaves = allstaves.drop('score', 1)

noises = allstaves.index.droplevel('method').droplevel('doc')
badnoise = noises[(allstaves.runs < 100).groupby(noises).any() ]
goodunl = ~unlabeled.index.get_level_values('noise').isin(badnoise)
unlabeled = unlabeled.iloc[goodunl]

allstaves = unlabeled[unlabeled.index.get_level_values('staffpage') != 'page']
allstaves.index = allstaves.index.droplevel('staffpage')
allstaves = allstaves.drop('score', 1)

allstaves = allstaves.groupby(allstaves.index).sum()
allstaves.index = pandas.MultiIndex.from_tuples(allstaves.index,
                                                names=pageMethods.index.names)
allstaves['score'] = allstaves.removed / allstaves.runs
uls = allstaves.score.fillna(0)

ulp = unlabeled.xs('page', level='staffpage').score.fillna(0)

sens = pandas.DataFrame([pageMethods.staff_sens, ulp]).T
sens = sens.ix[(~sens.isnull()).any(axis=1)]
sens.columns = ['labeled', 'unlabeled']
spec = pandas.DataFrame([pageMethods.staff_spec, uls]).T.fillna(0)
spec = spec.ix[(~spec.isnull()).any(axis=1)]
spec.columns = ['labeled', 'unlabeled']
f1 = (2*sens*spec/(sens + spec)).fillna(0)

def scatter_plot(score, color='.'):
    import pylab
    l, ul = np.array(score.T)
    pylab.plot(*((l, ul) + (color,) * (color is not None)))
