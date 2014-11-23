import zipfile
from PIL import Image
import numpy as np
from cStringIO import StringIO
import pylab
import moonshine.util
import sys, os
import cPickle
import sklearn.decomposition, sklearn.cluster
import datetime
pylab.ion()

N_IMAGES = 204800
PATCH_SIZE = 19
N_COMPONENTS = 128
WHITEN = False
N_CLUSTERS = 512
BATCH_SIZE = 1024*64

tic = datetime.datetime.now()

print 'Loading patches...'
trainset = zipfile.ZipFile('trainset_measures-current.zip')
images = np.array(trainset.infolist())
np.random.seed(42)
np.random.shuffle(images)
images = iter(images)

dots = np.rint(np.linspace(0, N_IMAGES-1, 80)[1:]).astype(int)

PATCHES_FILE = 'patches-%d.pkl' % N_IMAGES
if os.path.exists(PATCHES_FILE):
    patches = cPickle.load(open(PATCHES_FILE))
else:
    patches = np.zeros((N_IMAGES, PATCH_SIZE, PATCH_SIZE), np.uint8)
    for i in xrange(N_IMAGES):
        img = None
        while img is None:
            nextimg = images.next()
            #print nextimg.filename
            img = Image.open(StringIO(trainset.read(nextimg)))
            img = np.array(img)
            if (img.shape[0] < PATCH_SIZE*2 or img.shape[1] < PATCH_SIZE*4
                    or (img != 0).sum() <= PATCH_SIZE*5):
                img = None
                continue
            # We need to re-find the staves
            img_proj = (img != 0).sum(1)
            staff_pos = None
            for cutoff in [90, 95]:
                staff_cutoff = np.percentile(img_proj, cutoff)
                labels, num_labels = moonshine.util.label_1d(img_proj >= staff_cutoff)
                if num_labels == 5:
                    staff_pos = np.rint(moonshine.util.center_of_mass_1d(labels)).astype(int)
                    break
            if staff_pos is None:
                img = None
                continue
            patch = [0]
            MAXITER = 100
            it = 0
            staff_num = 0
            # Try to avoid patches with just staff
            # Expect less dark pixels on a staff than in between
            CUTOFFS = [PATCH_SIZE*2, PATCH_SIZE*4]
            while np.sum(np.where(patch, 1, 0)) <= CUTOFFS[staff_num % 2] and it < MAXITER:
                staff_num = np.random.randint(-1, 10)
                if staff_num % 2 == 0:
                    y0 = staff_pos[staff_num/2] - PATCH_SIZE/2
                elif staff_num == -1:
                    y0 = staff_pos[0] - 4 - PATCH_SIZE/2
                elif staff_num == 9:
                    y0 = staff_pos[-1] + 4 - PATCH_SIZE/2
                else:
                    sn = staff_num / 2
                    y0 = int(staff_pos[sn:sn+2].mean()) - PATCH_SIZE/2
                if y0 >= img.shape[0] - PATCH_SIZE:
                    continue
                x0 = np.random.randint(0, img.shape[1] - PATCH_SIZE + 1)
                patch = np.where(img[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE], 1, 0)
                it += 1
            if it == MAXITER:
                img = None
                continue
            patches[i] = patch
            if i in dots:
                sys.stderr.write('.')
    sys.stderr.write('\n')
    cPickle.dump(patches, open(PATCHES_FILE, 'w'))

print 'Computing principal components...'
PCA_FILE = "pca%d_%scomp%d" % (N_IMAGES, "w_" if WHITEN else "", N_COMPONENTS)
if os.path.exists(PCA_FILE):
    pca = cPickle.load(open(PCA_FILE))
    patchpca = pca.transform(patches.reshape((-1, PATCH_SIZE**2)))
else:
    pca = sklearn.decomposition.PCA(n_components=N_COMPONENTS, whiten=WHITEN)
    patchpca = pca.fit_transform(patches.reshape((-1, PATCH_SIZE**2)))
    cPickle.dump(pca, open(PCA_FILE, 'w'))

print 'Clustering...'
kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=BATCH_SIZE)
kmeans.fit(patchpca)
cluster_centers = pca.inverse_transform(kmeans.cluster_centers_)

cPickle.dump(kmeans.cluster_centers_,
             open('kmeans_comp%d_%sclust%d_batch%d.pkl'
                  % (N_COMPONENTS, 'whiten_' if WHITEN else '', N_CLUSTERS,
                     BATCH_SIZE),
                  'w'))

toc = datetime.datetime.now()
print 'Took', (toc-tic).total_seconds(), 'sec'
