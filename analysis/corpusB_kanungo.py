# Estimate Kanungo parameters for the Corpus B test set,
# and fit the single image quality grade using linear regression.
import env
import cPickle
import glob
import skimage.filter
import sklearn.ensemble
import numpy as np
import metaomr.page
import metaomr.kanungo
import pandas
import re
from PIL import Image

quality = pandas.DataFrame.from_csv('resources/corpusB_quality.csv')
results = pandas.DataFrame(columns='nu a0 a b0 b k'.split())
for filename in sorted(glob.glob('../corpusB/*/pdf/*.tif')):
    name = re.match(r'.+/(.+)\.tif', filename).group(1)
    image = np.array(Image.open(filename))
    if image.dtype is not bool:
        thresh = skimage.filter.threshold_otsu(image)
        image = image < thresh
    page = metaomr.page.Page(image)
    page.preprocess()
    if page.staff_dist is None:
        continue
    params = metaomr.kanungo.est_parameters(page)
    results.ix[name] = params.x
results = quality.join(results.clip(0, 5))
results.to_csv('results/corpusB_params.csv')
X = results[range(1, 7)]
Y = results[[0]]
Y = Y[~X.isnull().any(1)]
X = X[~X.isnull().any(1)]
X = np.array(X)
Y = np.array(Y)
leave1out = []
model = sklearn.ensemble.RandomForestRegressor()
for i in xrange(len(X)):
    x = np.r_[X[:i], X[i+1:]]
    y = np.r_[Y[:i], Y[i+1:]]
    model.fit(x,y)
    leave1out.append([Y[i, 0], model.predict(X[i])[0]])
model.fit(X, Y)
leave1out = np.array(leave1out)
cPickle.dump(dict(X=X, Y=Y, leave1out=leave1out, model=model), open('results/image_quality_fit.pkl','w'))
