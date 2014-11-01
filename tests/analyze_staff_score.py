import pandas
import numpy as np
import gzip
from glob import glob
from cStringIO import StringIO
import re

csvstr = StringIO()
csvstr.write(',removed,runs,score\n')
for filename in sorted(glob('staves/*.csv.gz')):
    fbase = re.search('IMSLP[0-9]+', filename).group(0)
    f = gzip.open(filename)
    f.readline()
    for line in f.readlines():
        csvstr.write(fbase + '-' + line)
    csvstr.write(f.read())
csvstr.seek(0)

results = pandas.DataFrame.from_csv(csvstr)
ul_index = pandas.Series(list(results.index)).str.extract(
    '^(?P<doc>IMSLP[0-9]+)-(?P<method>[a-z]+)P(?P<page>[0-9]+)S?(?P<staffpage>[0-9]+|page)$')
ul_multiindex = pandas.MultiIndex.from_tuples(ul_index.apply(tuple, axis=1), names=ul_index.columns)
results.index = ul_multiindex

allstaves = results[results.index.get_level_values('staffpage') != 'page']
allstaves.index = allstaves.index.droplevel('staffpage')
allstaves = allstaves.drop('score', 1)

allstaves = results[results.index.get_level_values('staffpage') != 'page']
allstaves.index = allstaves.index.droplevel('staffpage')
allstaves = allstaves.drop('score', 1)

allstaves = allstaves.groupby(allstaves.index).sum()
allstaves.index = pandas.MultiIndex.from_tuples(allstaves.index,
                                                names='doc method page'.split())
allstaves['score'] = allstaves.removed / allstaves.runs
staffscore = allstaves.score.fillna(0)

pagescore = results.xs('page', level='staffpage').score

f1score = 2*pagescore*staffscore/(pagescore + staffscore)

def methods_scatter(scores, m1, m2, color='.'):
    import pylab
    scores = scores[scores.index.get_level_values('method').isin([m1,m2])]
    scorearr = np.array(scores.unstack(1).fillna(0)).T
    pylab.plot(*tuple(scorearr) + (color,))

def density(vals, color=None):
    import pylab
    from scipy.stats import gaussian_kde
    dens = gaussian_kde(vals)
    xs = np.linspace(0,1,500)
    pylab.plot(*[xs, dens(xs)] + [color] * (color is not None))
