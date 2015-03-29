# Fit IMSLP user ratings to the estimated Kanungo degradation of the score.
import cPickle
import glob
import gzip
import re
import numpy as np
import pandas as pd
import sklearn.linear_model
import os.path
LEARNER = lambda: sklearn.linear_model.Lasso(alpha=0.01, normalize=True)
#LEARNER = sklearn.linear_model.LinearRegression

works = cPickle.load(open('imslp/imslp_works.pkl'))
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
feats = feats.join(deskew_feats, how='left')

#staffscore = sorted(glob.glob('results/imslp_staffscore/*.csv.gz'))
#staff_feats = pd.DataFrame()
#for f in staffscore:
#    result = pd.DataFrame.from_csv(gzip.open(f), index_col=range(3))
#    staff = result.index.get_level_values('staff')
#    result.index = result.index.droplevel('staff')
#    eachstaff = result.score.ix[staff != 'page']
#    staff_feats = staff_feats.append(pd.DataFrame(dict(
#                    staff_sens=result.score.ix[staff == 'page'],
#                    staff_spec=eachstaff.groupby(eachstaff.index).mean().fillna(0))))
#feats = feats.join(staff_feats, how='left')

ratings = pd.DataFrame(columns=['rating'])
regex = ''
for title, work in works.iteritems():
    for score in work['scores']:
        if 'rating' in score and score['rating']:
            score_id = os.path.basename(score['public_link'])
            ratings.ix[score_id] = (score['rating'],)
            regex += '|IMSLP' + score_id
ratings['id'] = ratings.index
regex = re.compile('^' + regex[1:] + '-')

if False:
    docfeats = feats.copy()
    docfeats.index = docfeats.index.get_level_values('doc')
    docfeats = docfeats.groupby(docfeats.index).median()
    doc_id = pd.Series(docfeats.index).str.extract('IMSLP([0-9]+)')
    doc_id.index = docfeats.index
    docfeats['id'] = doc_id
    docfeats = docfeats.merge(ratings, on='id', how='left').set_index(docfeats.index)
    docfeats = docfeats.ix[~ docfeats.isnull().any(1)]
    X = docfeats[range(14)]
    Y = docfeats['rating']
else:
    docs = pd.Series(kanungo.index.get_level_values('doc'), index=feats.index)
    feats['id'] = docs.str.extract('IMSLP([0-9]+)')
    feats = feats.merge(ratings, on='id', how='left').set_index(feats.index)
    feats = feats.ix[~ feats.isnull().any(1)]
    X = feats[range(12)]
    Y = feats['rating']

model = LEARNER()
model.fit(X, Y)
Ypred = model.predict(X)

Yperm = np.array(Y)
np.random.shuffle(Yperm)
nullmodel = LEARNER()
nullmodel.fit(X, Yperm)
