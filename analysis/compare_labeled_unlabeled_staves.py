import pandas
import numpy as np
import gzip
import pylab

labeled = pandas.DataFrame.from_csv(gzip.open('pagedata-staves.csv.gz'))
index = pandas.Series(list(labeled.index)).str.extract('^(?P<method>[a-z]+)-(?P<doc>[a-z0-9]+)-(?P<noise>.+)')
multiindex = pandas.MultiIndex.from_tuples([tuple(index.iloc[i]) for i in xrange(len(index))])
labeled.index = multiindex
pageMethods = labeled.fillna(0)
pageMethods.index.names = 'method doc noise'.split()

unlabeled = pandas.DataFrame.from_csv(gzip.open('staves.csv.gz'))
ul_index = pandas.Series(list(unlabeled.index)).str.extract(
    '^(?P<method>[a-z]+)-?(?P<native>native)?-(?P<doc>[a-z0-9]+)-(?P<noise>.+?)-?(?P<staffpage>S[0-9]+|page)$')
ul_index.native.fillna('', inplace=True)
ul_multiindex = pandas.MultiIndex.from_tuples(ul_index.apply(tuple, axis=1), names=ul_index.columns)
unlabeled.index = ul_multiindex

# Remove noise conditions which remove too many of the runs
dummy = unlabeled.xs('dummy')
runsmean = dummy.runs.groupby(map(dummy.index.get_level_values,
                                  ['noise', 'doc'])).mean()
runsremoved = runsmean.unstack(0).copy()
runs_orig = runsmean.xs('orig')
for col in runsremoved:
    runsremoved[col] /= runs_orig
toonoisy = runsremoved.min(axis=0) < 0.25
oknoise = toonoisy[~toonoisy].index
unlabeled = unlabeled[unlabeled.index.get_level_values('noise').isin(oknoise)]

ul_methods = unlabeled.index.get_level_values('method')
to_analyze = np.array(ul_methods != 'dummy')
unlabeled = unlabeled.iloc[to_analyze]

# Remove duplicates
unlabeled = unlabeled.groupby(unlabeled.index).first()
unlabeled.index = pandas.MultiIndex.from_tuples(unlabeled.index, names=ul_index.columns)

# Must replace non-native with native
unlabeled.index = unlabeled.index.droplevel('native')

allstaves = unlabeled[unlabeled.index.get_level_values('staffpage') != 'page']
allstaves.index = allstaves.index.droplevel('staffpage')
allstaves = allstaves.drop('score', 1)

#noises = allstaves.index.droplevel('method').droplevel('doc')
#badnoise = noises[(allstaves.runs < 100).groupby(noises).any()]
#goodunl = ~unlabeled.index.get_level_values('noise').isin(badnoise)
#unlabeled = unlabeled.iloc[goodunl]

allstaves = unlabeled[unlabeled.index.get_level_values('staffpage') != 'page']
allstaves.index = allstaves.index.droplevel('staffpage')
allstaves = allstaves.drop('score', 1)

allstaves = allstaves.groupby(allstaves.index).sum()
allstaves.index = pandas.MultiIndex.from_tuples(allstaves.index,
                                                names=pageMethods.index.names)
allstaves['score'] = allstaves.removed / allstaves.runs
uls = allstaves.score.fillna(0)

ulp = unlabeled.xs('page', level='staffpage').score
ulp = ulp.groupby(ulp.index).first()
ulp.index = pandas.MultiIndex.from_tuples(ulp.index, names='method doc noise'.split())

sens = pandas.DataFrame(pageMethods.staff_sens).join(pandas.DataFrame(ulp))
sens = sens[(~sens.isnull()).all(axis=1)]
sens.columns = ['labeled', 'unlabeled']
# print sens.unstack(0).fillna(0).mean(0).unstack(0)
spec = pandas.DataFrame(pageMethods.staff_spec).join(pandas.DataFrame(uls)).fillna(0)
spec = spec[(~spec.isnull()).all(axis=1)]
spec.columns = ['labeled', 'unlabeled']
f1 = (2*sens*spec/(sens + spec)).fillna(0)

def scatter_plot(score, color='.'):
    l, ul = np.array(score.T)
    pylab.plot(*((l, ul) + (color,) * (color is not None)))

scatter_plot(f1, 'b.')
pylab.xlim([0, 1])
pylab.ylim([0, 1])
pylab.xlabel('Labeled Staff F1 Score')
pylab.ylabel('Unlabeled Staff F1 Score')
# Linear regression: ul = m*l + b
l, ul = np.array(f1.T)
m, b = np.polyfit(l, ul, 1)
pylab.plot([0, 1], [b, b+m], 'r')
pylab.savefig('gamera_labeled_vs_unlabeled_scatter.png')
