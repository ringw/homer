import numpy as np
import pandas as pd
import skimage.measure
import scipy.optimize
from metaomr import bitimage

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
    for page in pages:
        for i, staff in enumerate(page.bars):
            for j, staffmeasure in enumerate(staff):
                moments.append(np.concatenate(map(hu_moments, staffmeasure)))
                cols.append((title, measure, page, staff, staffmeasure))
                measure += 1
    cols = pd.MultiIndex.from_tuples(cols, levels='score measure page staff staffmeasure'.split())
    moments = pd.DataFrame(moments, index=cols)
    return moments
