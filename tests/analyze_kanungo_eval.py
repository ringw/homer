import pandas as pd
import numpy as np

results = pd.DataFrame.from_csv('kanungo_eval.csv', index_col=range(4), header=range(2))

# Clip estimates
for noise in 'nu a0 b0'.split():
    results['estimate', noise] = results['estimate', noise].clip(0, 1)
for const in list('abk'):
    results['estimate', const] = results['estimate', const].clip(0, 5)
results['estimate', 'k'] = np.rint(results['estimate', 'k'])

#chisq = results.estimate.stat.xs('chisq',level='test').unstack('method')
#ks = results.estimate.stat.xs('ks',level='test').unstack('method')

allvals = results.stack(0)
means = results.real.mean(0)
stds = results.real.std(0)
def zscore(xs):
    if xs.name[1] in means:
        return (xs - means[xs.name[1]]) / stds[xs.name[1]]
    else:
        return xs
results_norm = results.apply(zscore, axis=0)

diff = (results_norm.real - results_norm.estimate)['nu a0 a b0 b k'.split()].abs()
errors = diff.sum(1)
#method_error = errors.unstack('method').mean(0)
#chisq_error = errors.xs('chisq',level='test').unstack('method').mean(0)
#ks_error = errors.xs('ks',level='test').unstack('method').mean(0)

# Un-normalized error
error_real = (results.real-results.estimate)['nu a0 a b0 b k'.split()].abs()
