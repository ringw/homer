from .gpu import *
from .staffboundary import distance_transform_kernel
from pyopencl import LocalMemory
from . import bitimage, staffsize
from scipy.optimize import minimize
import glob
import moonshine

prg = build_program(['clrand', 'kanungo'])

def normalize_page(page):
    staffsize.staffsize(page)
    return bitimage.scale(page.img, 8.0 / page.staff_dist, align=512)

IDEAL_SET = None
def load_ideal_set():
    global IDEAL_SET
    if IDEAL_SET is None:
        ideal_imgs = glob.glob('resources/ideal_set/*.png')
        IDEAL_SET = []
        for img in ideal_imgs:
            page, = moonshine.open(img)
            IDEAL_SET.append(normalize_page(page))
    return IDEAL_SET

# Source: T. Kanungo, R. M. Haralick, and I. Phillips. "Global and local
# document degradation models." In Proceedings of the Second International
# Conference on Document Analysis and Recognition, pages 730-734. IEEE, 1993.
def degrade(img, params):
    # Values from scipy.optimize are real-valued and potentially negative
    params = np.array(params)
    params[[0, 1, 3]] = np.clip(params[[0, 1, 3]], 0, 1)
    params[[2, 4]] = np.clip(params[[2, 4]], 0, np.inf)
    params = list(params)
    params[5] = max(0, np.rint(params[5]).astype(int))
    img = noise(img, params)
    img = closing(img, params)
    return img
def noise(img, (nu, a0, a, b0, b, k)):
    new_img = img.copy()
    img = img.get()
    fg_dist = thr.to_device(np.where(np.unpackbits(img),0,99).astype(np.int32))
    distance_transform_kernel(fg_dist, 10)
    bg_dist = thr.to_device(np.where(np.unpackbits(~img),0,99).astype(np.int32))
    distance_transform_kernel(bg_dist, 10)
    seed = thr.to_device(np.random.randint(1, 2**32, 2).astype(np.uint32))
    prg.kanungo_noise(new_img, fg_dist, bg_dist,
                      np.float32(nu),
                      np.float32(a0), np.float32(a),
                      np.float32(b0), np.float32(b),
                      seed,
                      LocalMemory(8),
                      global_size=new_img.shape[::-1])
    return new_img
def closing(img, (nu, a0, a, b0, b, k)):
    return bitimage.closing(img, numiter=k)

# Source: Kanungo, Tapas, and Qigong Zheng. "Estimation of morphological
# degradation model parameters." In 2001 IEEE International Conference on
# Acoustics, Speech, and Signal Processing. Vol. 3. IEEE, 2001.
def pattern_hist(img):
    patterns = thr.empty_like(Type(np.uint16, (img.shape[0], img.shape[1]*8)))
    prg.patterns_3x3(img, patterns, global_size=img.shape[::-1])
    hist = np.bincount(patterns.get().ravel()).copy()
    hist.resize(512)
    return hist

def test_hists(hist1, hist2):
    cdf1 = np.cumsum(hist1).astype(float) / hist1.sum()
    cdf2 = np.cumsum(hist2).astype(float) / hist2.sum()
    ks = np.abs(cdf1 - cdf2).max()
    return ks

def est_parameters(img, ideal_set=None):
    if ideal_set is None:
        ideal_set = load_ideal_set()
    img_hist = pattern_hist(img)
    def objective(params):
        degraded = [degrade(ideal_img, params) for ideal_img in ideal_set]
        hists = [pattern_hist(degraded_img) for degraded_img in degraded]
        combined_hist = np.sum(hists, axis=0)
        return test_hists(img_hist, combined_hist)
    params_0 = np.array([0.01, 0.01, 1, 0.01, 1, 1])
    return minimize(objective, params_0, method='nelder-mead', options=dict(xtol=1e-3, disp=True))
