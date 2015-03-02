import numpy as np
import pandas as pd
import skimage.measure
import scipy.optimize
import scipy.spatial.distance as ssd
from metaomr import bitimage
from glob import glob
import os.path

def hu_moments(measure):
    img = bitimage.as_hostimage(measure.get_image()).astype(float)
    m = skimage.measure.moments(img)
    cr = m[0,1] / m[0,0]
    cc = m[1,0] / m[0,0]
    mu = skimage.measure.moments_central(img, cr, cc)
    nu = skimage.measure.moments_normalized(mu)
    hu = skimage.measure.moments_hu(nu)
    return hu

def score_measure_moments(pages, title=None):
    moments = []
    cols = []
    measure = 0
    for p, page in enumerate(pages):
        if not hasattr(page, 'bars'):
            continue
        for i, staff in enumerate(page.bars):
            for j, staffmeasure in enumerate(staff):
                moments.append(np.concatenate(map(hu_moments, staffmeasure)))
                cols.append((title, measure, p, i, j))
                measure += 1
    cols = pd.MultiIndex.from_tuples(cols, names='score measure page staff staffmeasure'.split())
    moments = pd.DataFrame(moments, index=cols)
    return moments

def measure_cov(measures_path):
    files = sorted(glob(measures_path + '/*.csv'))
    docs = [pd.DataFrame.from_csv(f, index_col=range(5)) for f in files]
    all_measures = pd.concat(docs)
    staff_ind = [map(str, range(7*i, 7*(i+1))) for i in xrange(all_measures.shape[1] / 7)]
    measures_staff = [all_measures[ind] for ind in staff_ind]
    cols = ['hu%d' % i for i in range(7)]
    for staff in measures_staff:
        staff.columns = cols
    flat_measures = pd.concat(measures_staff)
    flat_measures = flat_measures.ix[(~flat_measures.isnull()).all(1)]
    return np.cov(flat_measures.T)
MEASURE_COV_PATH = 'results/measure_hu_cov.csv'
if os.path.exists(MEASURE_COV_PATH):
    MEASURE_COV = np.loadtxt(MEASURE_COV_PATH)
else:
    MEASURE_COV = measure_cov('imslp/measures')
    np.savetxt(MEASURE_COV_PATH, MEASURE_COV)

def align_docs(doc1, doc2, gap_penalty=100):
    if doc1.shape[1] != doc2.shape[1] or min(doc1.shape[0], doc1.shape[0]) <= 1:
        return None
    num_parts = doc1.shape[1] / 7
    # Block covariance matrix for multiple parts
    blocks = []
    for i in xrange(num_parts):
        blocks.append([np.zeros((7,7))] * i + [MEASURE_COV] + [np.zeros((7,7))] * (num_parts - i - 1))
    V = np.bmat(blocks)
    VI = np.linalg.inv(V)
    dists = ssd.cdist(doc1, doc2, 'mahalanobis', VI)
    scores = np.empty((doc1.shape[0], doc2.shape[0]))
    scores[0, 0] = dists[0, 0]
    for i in xrange(doc1.shape[0]):
        scores[i, 0] = i * gap_penalty
    for j in xrange(doc2.shape[0]):
        scores[0, j] = j * gap_penalty
    dx = np.array([-1, -1, 0], int)
    dy = np.array([-1, 0, -1], int)
    ptr = np.empty_like(scores, int)
    ptr[0, 0] = 0
    ptr[1:, 0] = 2
    ptr[0, 1:] = 1
    for i in xrange(1, doc1.shape[0]):
        for j in xrange(1, doc2.shape[0]):
            new_scores = scores[i + dy, j + dx]
            new_scores[0] += dists[i, j]
            new_scores[1:] += gap_penalty
            ptr[i, j] = np.argmin(new_scores)
            scores[i, j] = new_scores[ptr[i, j]]
    score = scores[i, j]
    alignment = []
    while i >= 0 and j >= 0:
        direction = ptr[i, j]
        alignment.append((i if direction != 1 else -1,
                          j if direction != 2 else -1))
        i += dy[direction]
        j += dx[direction]
    return alignment[::-1], score
