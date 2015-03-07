import numpy as np
import pandas as pd
import skimage.measure
import scipy.optimize
import scipy.spatial.distance as ssd
from metaomr import bitimage
from glob import glob
import os.path
import scipy.misc

def measure_projections(page, title=None, p=0, m=0):
    all_measures = []
    cols = []
    measure = m
    if not hasattr(page, 'systems'):
        return pd.DataFrame()
    for i, system in enumerate(page.systems):
        nb = len(system['barlines']) - 1
        measures = [[] for _ in xrange(nb)]
        cols += [(title, measure + k, p, i, k) for k in xrange(nb)]
        for j, staffnum in enumerate(xrange(system['start'], system['stop'] + 1)):
            staff = bitimage.as_hostimage(page.staves.extract_staff(staffnum)).astype(bool)
            for k in xrange(nb):
                measure_img = staff[:, int(system['barlines'][k,:,0].mean())
                                        : int(system['barlines'][k+1,:,0].mean())]
                scaled_measure = scipy.misc.imresize(measure_img, (5*6, int(30.0 * measure_img.shape[1] / measure_img.shape[0])))
                measures[k] += list(scaled_measure.sum(1))
        all_measures += measures
        measure += nb
    if len(cols):
        cols = pd.MultiIndex.from_tuples(cols, names='score measure page staff staffmeasure'.split())
        all_measures = pd.DataFrame(all_measures, index=cols)
        return all_measures
    else:
        return pd.DataFrame()

def measure_cov(measures_path):
    NUM_FEATS = 30
    files = sorted(glob(measures_path + '/*.csv'))
    docs = [pd.DataFrame.from_csv(f, index_col=range(5)) for f in files]
    all_measures = pd.concat(docs)
    staff_ind = [map(str, range(NUM_FEATS*i, NUM_FEATS*(i+1))) for i in xrange(all_measures.shape[1] / NUM_FEATS)]
    measures = []
    for f in files:
        doc = pd.DataFrame.from_csv(f, index_col=range(5))
        for i in xrange(doc.shape[1] / NUM_FEATS):
            measures.append(np.array(doc[doc.columns[i*NUM_FEATS:(i+1)*NUM_FEATS]]))
    measures = np.concatenate(measures)
    measures /= measures.sum(1)[:, None]
    # Normalize to 10th and 90th percentiles
    scale = np.percentile(measures, 0.9, axis=0) - np.percentile(measures, 0.1, axis=0)
    measures /= scale[None, :]
    measures = measures[~np.isnan(measures).any(1)]
    return np.cov(measures.T), scale
MEASURE_COV_PATH = 'results/measure_proj_cov.npz'
if os.path.exists(MEASURE_COV_PATH):
    with np.load(MEASURE_COV_PATH) as measure_data:
        MEASURE_COV = measure_data['cov']
        MEASURE_SCALE = measure_data['scale']
else:
    MEASURE_COV, MEASURE_SCALE = measure_cov('imslp/measures')
    np.savez(MEASURE_COV_PATH, cov=MEASURE_COV, scale=MEASURE_SCALE)

def align_docs(doc1, doc2, gap_penalty=10):
    if doc1.shape[1] != doc2.shape[1] or min(doc1.shape[0], doc1.shape[0]) <= 1:
        return None
    doc1 = np.array(doc1, float)
    doc2 = np.array(doc2, float)
    NUM_FEATS = 30
    num_parts = doc1.shape[1] / NUM_FEATS
    # Block covariance matrix for multiple parts
    blocks = []
    for i in xrange(num_parts):
        blocks.append([np.zeros((NUM_FEATS,NUM_FEATS))] * i
                        + [MEASURE_COV]
                        + [np.zeros((NUM_FEATS,NUM_FEATS))] * (num_parts - i - 1))
        # Normalize each measure
        feats = slice(i*NUM_FEATS, (i+1)*NUM_FEATS)
        doc1[:, feats] /= doc1[:, feats].sum(1)[:, None]
        doc2[:, feats] /= doc2[:, feats].sum(1)[:, None]
    V = np.bmat(blocks)
    VI = np.linalg.inv(V)
    scaled1 = doc1 / np.repeat(MEASURE_SCALE, num_parts)
    scaled2 = doc2 / np.repeat(MEASURE_SCALE, num_parts)
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
                          j if direction != 2 else -1,
                          dists[i, j] if direction == 0 else gap_penalty))
        i += dy[direction]
        j += dx[direction]
    alignment = alignment[::-1]
    return pd.DataFrame(alignment, columns='doc1 doc2 score'.split())
