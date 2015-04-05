import pandas as pd
import numpy as np
import glob
import gzip
import zipfile
import os
import re
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR

align = pd.DataFrame.from_csv('results/beethoven_omr.csv', index_col=range(2))
align_omrfile = pd.Series(align.index.get_level_values('omr'), index=align.index)
align['id'] = align_omrfile.str.extract('(IMSLP[0-9]+)')

sonatas = pd.DataFrame.from_csv('resources/beethoven_sonatas.csv',
                                header=None)

kanungo = pd.DataFrame()
for f in sorted(glob.glob('results/imslp_kanungo/*.csv')):
    results = pd.DataFrame.from_csv(f, index_col=range(3))
    kanungo = kanungo.append(results.xs('ks', level='fn'))
kanungo = kanungo['nu a0 a b0 b k'.split()]
kanungo['nu'] = kanungo['nu'].clip(0, 1)
kanungo['a0'] = kanungo['a0'].clip(0, 1)
kanungo['a'] = kanungo['a'].clip(0, np.inf)
kanungo['b0'] = kanungo['b0'].clip(0, 1)
kanungo['b'] = kanungo['b'].clip(0, np.inf)
kanungo['k'] = np.rint(kanungo['k'].clip(0, 5))
kanungo['a0a'] = (kanungo['a0'] * kanungo['a']).fillna(0)
kanungo['b0b'] = (kanungo['b0'] * kanungo['b']).fillna(0)
feats = kanungo.copy()

deskew = sorted(glob.glob('results/imslp_deskew/*.csv.gz'))
deskew_feats = pd.DataFrame()
for f in deskew:
    result = pd.DataFrame.from_csv(gzip.open(f), index_col=range(2))
    left = (result.staff_left / 32 + 1).round()
    right = (result.staff_right / 32 - 1).round()
    skews = result[range(4, result.shape[1])]
    skews.columns = range(skews.shape[1])
    skews = skews.apply(lambda x: x[int(left.ix[x.name]):int(right.ix[x.name])],
                        axis=1)
    skewdiff = skews.T.diff().T.abs()
    deskew_feats = deskew_feats.append(pd.DataFrame(dict(
                      mean_skew_norm=skewdiff.mean(1) / result.staff_dist,
                      median_skew_norm=skewdiff.median(1) / result.staff_dist,
                      staff_dist=result.staff_dist,
                      staff_thick_ratio=result.staff_thick / result.staff_dist)))
feats = feats.join(deskew_feats, how='left').fillna(0)
feats = feats[(feats != 0).any(1)]

featdoc = pd.Series(feats.index.get_level_values('doc'), index=feats.index)
featdoc = featdoc.str.extract('(IMSLP[0-9]+)')

# Split page features by movement
midiname = pd.Series(index=feats.index)
docnames = midiname.index.get_level_values('doc')
for mvmtfile in glob.glob('movements/*.zip'):
    name = os.path.basename(os.path.splitext(mvmtfile)[0])
    if name not in docnames:
        continue
    imslpid = re.search('IMSLP[0-9]+', name).group(0)
    sonatanum = sonatas.ix[imslpid][1]
    z = zipfile.ZipFile(mvmtfile)
    for f in z.namelist():
        m = re.match(r'mvmt([0-9])/page([0-9]+)\.tif', f)
        fakemidi = '%s_beet%d_%d.mid' % (imslpid, sonatanum, int(m.group(1))+1)
        page = int(m.group(2))
        if (name, page) in midiname.index:
            midiname[(name, page)] = fakemidi

mvmtfeats = feats.groupby(midiname).mean()
mvmtrealfile = [x.split('_',1)[1] for x in mvmtfeats.index]
mvmtfeats.index = pd.MultiIndex.from_tuples(zip(mvmtrealfile, mvmtfeats.index),
                                            names=['real','omr'])
results = align.join(mvmtfeats)
results = results[~ results.isnull().any(1)]

X = results['nu a0 a b0 b k mean_skew_norm staff_dist staff_thick_ratio'.split()]
# Normalize X by quantiles
X -= X.quantile(0.1)
X /= X.quantile(0.9)

Y = results['F1']

docs = align.index.get_level_values('real').unique()
Ybest = Y.unstack(1).idxmax(1)
Ypred = pd.Series(index=Ybest.index, name='pred')
for doc in docs:
    istrain = np.ones(len(docs), bool)
    istrain[list(docs).index(doc)] = 0
    train = pd.Series(index=align.index)
    train[:] = True
    train[doc] = False

    model = SVR()
    model.fit(X[train], Y[train])

    Ytest = Y[~train]
    pred = pd.Series(model.predict(X[~train]), index=Ytest.index)
    predbest = pred.unstack(1).idxmax(1)
    Ypred[predbest.index[0]] = predbest[0]
