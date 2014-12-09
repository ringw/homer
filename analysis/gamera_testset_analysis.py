import pandas
import numpy as np
import gzip

labeled = pandas.DataFrame.from_csv(gzip.open('pagedata-staves.csv.gz'))
index = pandas.Series(list(labeled.index)).str.extract('^(?P<method>[a-z_0-9]+)-(?P<doc>[a-z0-9]+)-(?P<noise>.+)')
multiindex = pandas.MultiIndex.from_tuples([tuple(index.iloc[i]) for i in xrange(len(index))])
labeled.index = multiindex
labeled = labeled.fillna(0)
labeled.index.names = 'method doc noise'.split()
labeled.columns = 'sens spec time'.split()
labeled = labeled.ix[~pandas.Series(labeled.index.get_level_values('method'),index=labeled.index).str.match('miyao|carter|hough_1deg|roach')]
labeled['method2']=pandas.Series(labeled.index.get_level_values('method'),index=labeled.index).str.replace('hough_pi250_201','hough')
labeled.set_index('method2',append=True,inplace=True)
labeled.index = labeled.index.droplevel('method')
labeled.index = labeled.index.reorder_levels(['method2','doc','noise'])
labeled.index.names = ['method','doc','noise']

unlabeled = pandas.DataFrame.from_csv(gzip.open('staves.csv.gz'))
ul_index = pandas.Series(list(unlabeled.index)).str.extract(
    '^(?P<method>[a-z_0-9]+)-?(?P<native>native)?-(?P<doc>[a-z0-9]+)-(?P<noise>.+?)-?(?P<staffpage>S[0-9]+|page)$')
ul_index.native.fillna('', inplace=True)
ul_multiindex = pandas.MultiIndex.from_tuples(ul_index.apply(tuple, axis=1), names=ul_index.columns)
unlabeled.index = ul_multiindex
unlabeled = unlabeled.ix[~pandas.Series(unlabeled.index.get_level_values('method'),index=unlabeled.index).str.match('miyao|carter|hough_1deg|roach')]
unlabeled['method2']=pandas.Series(map(str,unlabeled.index.get_level_values('method')),index=unlabeled.index).str.replace('hough_pi250_201','hough')
unlabeled.set_index('method2',append=True,inplace=True)
unlabeled.index = unlabeled.index.droplevel('method')
unlabeled.index = unlabeled.index.reorder_levels(['method2','native','doc','noise','staffpage'])
unlabeled.index.names = ['method','native','doc','noise','staffpage']

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
allstaves = allstaves.groupby(allstaves.index).sum()
allstaves.index = pandas.MultiIndex.from_tuples(allstaves.index,
                                                names=labeled.index.names)
allstaves['score'] = allstaves.removed / allstaves.runs
ul_sens = allstaves.score.fillna(0)
ul_spec = unlabeled.xs('page', level='staffpage').score
ul_spec = ul_spec.groupby(ul_spec.index).first()
ul_spec.index = pandas.MultiIndex.from_tuples(ul_spec.index, names='method doc noise'.split())
unlabeled = pandas.DataFrame(ul_sens).join(pandas.DataFrame(ul_spec))
unlabeled.columns = ['sens', 'spec']
scores = labeled[['sens','spec']].join(unlabeled,rsuffix='_ul')
scores.columns = pandas.MultiIndex.from_product([
                        ['labeled','unlabeled'],
                        ['sens','spec']],
                    names=['type','score'])
scores.fillna(0, inplace=True)

sens = scores[[0,2]]
sens.columns = ['labeled', 'unlabeled']
spec = scores[[1,3]]
spec.columns = ['labeled', 'unlabeled']
f1 = (2*sens*spec/(sens + spec)).fillna(0)

def scatter_plot(score, color='.'):
    import pylab
    l, ul = np.array(score.T)
    pylab.plot(*((l, ul) + (color,) * (color is not None)))

def scatter_summary(m):
    import pylab
    m = m.unstack('method').mean(0)
    methods = m.spec.index
    for method in methods:
        pylab.plot(m.spec.ix[method], m.sens.ix[method], 'o')
    pylab.legend(methods, numpoints=1, loc='lower left')
    pylab.xlim([0.7,1])
    pylab.ylim([0.7,1])

def scatter_method(m):
    import pylab
    m = m.unstack('method').mean(0)
    methods = m.spec.index
    for method in methods:
        pylab.plot(m.spec.ix[method], m.sens.ix[method], 'o')
    pylab.legend(methods, numpoints=1, loc='lower left')
    pylab.xlim([0.7,1])
    pylab.ylim([0.7,1])

if __name__ == '__main__':
    import pylab
    pylab.figure()
    scatter_plot(f1, 'b.')
    pylab.xlim([0, 1])
    pylab.ylim([0, 1])
    pylab.xlabel('Labeled Staff F1 Score')
    pylab.ylabel('Unlabeled Staff F1 Score')
    # Linear regression: ul = m*l + b
    l, ul = np.array(f1.T)
    m, b = np.polyfit(l, ul, 1)
    pylab.plot([0, 1], [b, b+m], 'r')
    pylab.savefig('gamera_labeled_vs_unlabeled_scatter.pdf')

    scores.groupby(scores.index.get_level_values('method')).apply(lambda x: x.mean(axis=0)).to_csv('gamera_performance.csv')
