import env
import numpy as np
import metaomr
import metaomr.kanungo as kan
from metaomr.page import Page
import glob
import pandas as pd
import itertools
import os.path
import sys
from random import random, randint
IDEAL = [path for path in sorted(glob.glob('testset/modern/*.png'))
              if 'nostaff' not in path]

NU = np.linspace(0, 0.05, 3)
A0 = B0 = np.linspace(0, 0.2, 3)
A = B = np.linspace(0.5, 2.5, 3)
K = np.linspace(0, 3, 4)

def random_params():
    if random() < 0.25:
        nu = 0
    else:
        nu = random() * 0.05
    if random() < 0.25:
        a0 = a = 0
    else:
        a0 = random() * 0.2
        a = 0.5 + random() * 2
    if random() < 0.25:
        b0 = b = 0
    else:
        b0 = random() * 0.2
        b = 0.5 + random() * 2
    k = randint(0, 4)
    return nu, a0, a, b0, b, k

columns = pd.MultiIndex.from_product([['real', 'estimate'], 'nu a0 a b0 b k stat'.split()])
cols = []
results = []
for image in IDEAL:
    name = os.path.basename(image).split('.')[0]
    page, = metaomr.open(image)
    kimg = kan.KanungoImage(kan.normalized_page(page)[0])
    #for params in itertools.product(NU,A0,A,B0,B,K):
    for i in xrange(100):
        params = random_params()
        synth = Page(kimg.degrade(params))
        synth.staff_dist = 8
        for fun in ['ks', 'chisq']:
            for method in 'Nelder-Mead Powell Anneal'.split():
                est_params = kan.est_parameters(synth, test_fn=kan.test_hists_ks if fun == 'ks' else kan.test_hists_chisq, opt_method=method)
                cols.append((name, fun, method) + tuple(params))
                results.append(list(params) + [0] + list(est_params.x) + [est_params.fun])
    res = pd.DataFrame(results, columns=columns)
    res.index = pd.MultiIndex.from_tuples(cols)
    res.index.names = 'doc test method nu a0 a b0 b k'.split()
    res.to_csv('kanungo_eval.csv')
