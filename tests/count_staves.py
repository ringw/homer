import numpy as np
import pandas

orig_staves = pandas.DataFrame.from_csv('staves.csv')
# str.extract doesn't work on index.to_series() ???
names = pandas.Series(list(orig_staves.index.to_series()))
our_index = names.str.extract('^(?P<method>[a-z]+(?:-native)?)-(?P<doc>[a-z0-9]+)-(?P<noise>[a-z0-9\.\-]+?)-(?P<staff>[A-z0-9]+)$')
our_index.index = orig_staves.index
staff_index = our_index[our_index.staff != 'page']
staff_multiindex = pandas.MultiIndex.from_arrays([staff_index.method, staff_index.doc, staff_index.noise])
staves = orig_staves[our_index.staff != 'page']
sgb = staves.groupby(staff_multiindex)
counter = pandas.Series(np.ones(len(staves.index),int), index=staves.index)
staff_count = counter.groupby(staff_multiindex).sum()
staff_count.index = pandas.MultiIndex.from_tuples(staff_count.index)
staff_count = pandas.DataFrame(staff_count).unstack()
staff_count.columns = staff_count.columns.levels[1]
staff_count.orig /= 2 # ???

counts = staff_count.ix['dummy']['orig']
methods = our_index.method.unique()
noises = our_index.noise.unique()
scores = pandas.DataFrame(dict((n, dict((m,(staff_count.ix[(m,n)] == counts).mean())
                                        for m in methods))
                               for n in noises))
