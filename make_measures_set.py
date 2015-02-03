import cStringIO
import gc
import metaomr.bitimage
import metaomr.image
import metaomr.page
import metaomr.staffsize
import numpy as np
import os
import re
import shutil
import zipfile
from PIL import Image

last_doc, last_page = None, None
if os.path.exists('trainset_measures.zip'):
    z = zipfile.ZipFile('trainset_measures.zip')
    path = z.infolist()[-1].filename
    last_doc = re.search('IMSLP[0-9]+', path).group(0)
    last_page = int(re.search('P([0-9]+)', path).group(1))

i = 0
out = zipfile.ZipFile('trainset_measures.zip', 'a', zipfile.ZIP_DEFLATED, True)
runs = open('runlength.csv', 'a')
for filename in open('trainpdfs').readlines():
    filename = filename.strip()
    imslpid = re.search('IMSLP[0-9]+', filename).group(0)
    if last_doc is not None and last_doc != imslpid:
        continue
    images = metaomr.image.read_pages(filename)
    if not (0 < len(images) <= 100):
        continue
    for pagenum, image in enumerate(images):
        if last_page is not None:
            if last_page == pagenum:
                last_doc, last_page = None, None
            continue
        print imslpid, pagenum
        page = metaomr.page.Page(image)
        page.preprocess()
        if type(page.staff_dist) is not int:
            continue
        dist = metaomr.staffsize.staff_dist_hist(page)
        thick = metaomr.staffsize.staff_thick_hist(page)
        print >> runs, ','.join([imslpid, str(pagenum)]
                                + map(str, dist) + map(str, thick))
        try:
            if page.staff_dist < 8 or not (6 <= len(page.staves()) <= 24):
                continue
            page.process()
            scale = 8.0 / page.staff_dist
            for s, system in enumerate(page.bars):
                for m, measure in enumerate(system):
                    for p, part in enumerate(measure):
                        img = part.get_image()
                        img = metaomr.bitimage.scale(img, scale)
                        out_image = metaomr.bitimage.as_hostimage(img)
                        if not out_image.any():
                            continue
                        # Clean up by removing margin
                        import metaomr.util
                        labels, nl = metaomr.util.label_1d(out_image.sum(1))
                        heights = np.bincount(labels)
                        label_id = heights[1:].argmax() + 1
                        label_range, = np.where(labels == label_id)
                        ymin = max(0, label_range[0] - page.staff_dist*2)
                        ymax = min(img.shape[0], label_range[-1] + page.staff_dist*2)
                        out_image = Image.fromarray(out_image[ymin:ymax]*255)
                        imageio = cStringIO.StringIO()
                        out_image.save(imageio, 'png')
                        imageio.seek(0)
                        out.writestr('%s-P%02dS%02dM%02dP%02d.png'
                                     % (imslpid, pagenum, s, m, p),
                                     imageio.read())
        except Exception, e:
            print e

        if i % 500 == 0:
            out.close()
            shutil.copy('trainset_measures.zip', 'trainset_measures-current.zip')
            out = zipfile.ZipFile('trainset_measures.zip', 'a', zipfile.ZIP_DEFLATED, True)
            runs.flush()
        del page
        gc.collect()
        pagenum += 1
        i += 1
