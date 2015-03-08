# Fit IMSLP user ratings to the estimated Kanungo degradation of the score.
import cPickle
import re
import numpy as np
import pandas as pd
import sklearn.svm
import os.path
LEARNER = sklearn.svm.SVR

works = cPickle.load(open('imslp/imslp_works.pkl'))
kanungo = pd.DataFrame.from_csv('results/imslp_kanungo.csv', index_col=range(2))
kanungo = kanungo['nu a0 a b0 b k'.split()]
kanungo['nu'] = kanungo['nu'].clip(0, 1)
kanungo['a0'] = kanungo['a0'].clip(0, 1)
kanungo['a'] = kanungo['a'].clip(0, np.inf)
kanungo['b0'] = kanungo['b0'].clip(0, 1)
kanungo['b'] = kanungo['b'].clip(0, np.inf)
kanungo['k'] = np.rint(kanungo['k'].clip(0, 5))

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
docs = pd.Series(kanungo.index.get_level_values('doc'), index=kanungo.index)
kanungo['id'] = docs.str.extract('IMSLP([0-9]+)')
kanungo = kanungo.merge(ratings, on='id', how='left').set_index(kanungo.index)
kanungo = kanungo.ix[~ kanungo.isnull().any(1)]
docs = pd.Series(kanungo.index.get_level_values('doc'), index=kanungo.index)
X = kanungo[range(6)]
Y = kanungo['rating']

model = LEARNER()
model.fit(X, Y)

leave1out = pd.DataFrame()
for doc in docs.unique():
    test = docs == doc
    l1o_model = LEARNER()
    l1o_model.fit(X.ix[~test], Y[~test])
    Yreal = Y[test]
    Ypred = l1o_model.predict(np.array(X.ix[test]))
    #leave1out.append((Yreal[0], Ypred))
    leave1out = leave1out.append(zip(Yreal, Ypred))
err = (leave1out[1] - leave1out[0]).abs()
print 'mean', err.mean(), 'median', err.median(), 'max', err.max()
