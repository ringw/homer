from .gpu import *
from .staffboundary import distance_transform_kernel
from pyopencl import LocalMemory
from . import bitimage, staffsize
from .page import Page
from scipy.optimize import minimize
import glob
import metaomr

prg = build_program(['kanungo'])

def normalized_page(page):
    if not hasattr(page, 'staff_dist'):
        staffsize.staffsize(page)
    scale = 8.0 / page.staff_dist
    img = bitimage.scale(page.img, scale, align=64)
    # Grab a 512x512 area around the center of the page
    h = int(page.orig_size[0] * scale)
    w = int(page.orig_size[1] * scale)
    img = bitimage.as_hostimage(img)[h/2-256:h/2+256, w/2-256:w/2+256]
    return bitimage.as_bitimage(img)
def normalized_staves(page):
    # Assume page is not rotated; scanned pages must be preprocessed first
    staffsize.staffsize(page)
    img = bitimage.scale(page.img, 8.0 / page.staff_dist, align=64)
    new_page = Page(img)
    staffsize.staffsize(new_page)
    num_staves = len(new_page.staves())
    return [new_page.staves.extract_staff(i) for i in xrange(num_staves)]

IDEAL_SET = None
def load_ideal_set():
    global IDEAL_SET
    if IDEAL_SET is None:
        ideal_imgs = glob.glob('resources/ideal_set/*.png')
        IDEAL_SET = []
        for img in ideal_imgs:
            page, = metaomr.open(img)
            IDEAL_SET.append(KanungoImage(normalized_page(page)))
    return IDEAL_SET

# Source: T. Kanungo, R. M. Haralick, and I. Phillips. "Global and local
# document degradation models." In Proceedings of the Second International
# Conference on Document Analysis and Recognition, pages 730-734. IEEE, 1993.
class KanungoImage:
    img = None
    fg_dist = None
    bg_dist = None
    def __init__(self, img):
        self.img = img
        img = img.get()
        self.fg_dist = thr.to_device(np.where(np.unpackbits(img),0,2**10).astype(np.int32))
        distance_transform_kernel(self.fg_dist, 10)
        self.bg_dist = thr.to_device(np.where(np.unpackbits(~img),0,2**10).astype(np.int32))
        distance_transform_kernel(self.bg_dist, 10)
    def degrade(self, params):
        # Values from scipy.optimize are real-valued and potentially negative
        params = np.array(params)
        params[[0, 1, 3]] = np.clip(params[[0, 1, 3]], 0, 1)
        params[[2, 4]] = np.clip(params[[2, 4]], 0, np.inf)
        params = list(params)
        params[5] = max(0, np.rint(params[5]).astype(int))
        img = self.noise(params)
        img = self.closing(img, params)
        return img
    def noise(self, (nu, a0, a, b0, b, k)):
        new_img = self.img.copy()
        prg.kanungo_noise(new_img, self.fg_dist, self.bg_dist,
                          np.float32(nu),
                          np.float32(a0), np.float32(a),
                          np.float32(b0), np.float32(b),
                          np.uint32(np.random.randint(0, 2**32)),
                          global_size=new_img.shape[::-1])
        return new_img
    def closing(self, img, (nu, a0, a, b0, b, k)):
        return bitimage.closing(img, numiter=k)

# Source: Kanungo, Tapas, and Qigong Zheng. "Estimation of morphological
# degradation model parameters." In 2001 IEEE International Conference on
# Acoustics, Speech, and Signal Processing. Vol. 3. IEEE, 2001.
# To reduce temporary memory usage, just return the list of patterns
# for each image, which can then be concatenated and bincounted
def pattern_list(img):
    patterns = thr.empty_like(Type(np.uint32, (img.shape[0], img.shape[1]*8)))
    prg.patterns_3x3(img, patterns, global_size=img.shape[::-1])
    return patterns.get().ravel()

def test_hists_ks(hist1, hist2):
    cdf1 = np.cumsum(hist1).astype(float) / hist1.sum()
    cdf2 = np.cumsum(hist2).astype(float) / hist2.sum()
    ks = np.abs(cdf1 - cdf2).max()
    p = None
    return ks, p
def test_hists_euc(hist1, hist2):
    # Normalize bins excluding all-white pattern
    hist1 = hist1.astype(float)
    hist2 = hist2.astype(float)
    hist1 = hist1[1:] / hist1[1:].sum()
    hist2 = hist2[1:] / hist2[1:].sum()
    return np.sqrt(np.sum((hist1 - hist2)[1:] ** 2)), None
import scipy.stats
test_hists_chisq = scipy.stats.chisquare

def est_parameters(page, ideal_set=None, opt_method='nelder-mead', test_fn=test_hists_chisq, maxfev=50):
    if ideal_set is None:
        ideal_set = load_ideal_set()
    page_center = normalized_page(page)
    patterns = pattern_list(page_center)
    page_hist = np.bincount(patterns).astype(np.int32).copy()
    page_hist.resize(2 ** (3*3))
    page_patterns = page_hist > 0
    page_patterns[0] = 0 # skip all white background
    page_freq = page_hist[page_patterns]
    page_freq = page_freq.astype(float) / page_freq.sum()
    def objective(params):
        degraded = [ideal_img.degrade(params) for ideal_img in ideal_set]
        patterns = np.concatenate([pattern_list(degraded_img) for degraded_img in degraded])
        combined_hist = np.bincount(patterns).astype(np.int32).copy()
        combined_hist.resize(2 ** (3*3))
        cmbf = combined_hist[page_patterns]
        cmbf = cmbf.astype(float) / cmbf.sum()
        res = test_fn(cmbf, page_freq)[0]
        return res
    minim_results = []
    for i in xrange(10):
        params_0 = np.array([0.01, 0.01, 0.5, 0.01, 0.5, 1]
                            + np.random.random(6)
                              * [0.09, 0.09, 3, 0.09, 3, 5])
        minim_results.append(minimize(objective, params_0, method=opt_method,
            options=dict(xtol=1e-4, maxfev=maxfev),
            bounds=[(0,0.5), (0,0.5), (0,10), (0,0.5), (0,10), (0,5)]))
    best_result = np.argmin([res.fun for res in minim_results])
    return minim_results[best_result], page_hist
