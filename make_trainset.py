import cStringIO
import gc
import moonshine.bitimage
import moonshine.image
import moonshine.page
import moonshine.staffsize
import re
import shutil
import zipfile
from PIL import Image

i = 0
out = zipfile.ZipFile('trainset.zip', 'w', zipfile.ZIP_DEFLATED, True)
runs = open('runlength.csv', 'a')
for filename in open('trainpdfs').readlines():
    filename = filename.strip()
    imslpid = re.search('IMSLP[0-9]+', filename).group(0)
    images = moonshine.image.read_pages(filename)
    if not (0 < len(images) <= 100):
        continue
    for pagenum, image in enumerate(images):
        print imslpid, pagenum
        page = moonshine.page.Page(image)
        page.preprocess()
        dark = moonshine.staffsize.dark_runs(page.img)
        light = moonshine.staffsize.light_runs(page)
        print >> runs, ','.join([imslpid, str(pagenum)]
                                + map(str, dark) + map(str, light))
        try:
            if (type(page.staff_dist) is not int or page.staff_dist < 8
                or not (6 <= len(page.staves()) <= 24)):
                continue
            scale = 8.0 / page.staff_dist
            img = moonshine.bitimage.scale(page.img, scale)
            ns = moonshine.bitimage.scale(page.staves.nostaff(), scale)
            out_image = moonshine.bitimage.as_hostimage(img) + moonshine.bitimage.as_hostimage(ns)
            sh = int(page.orig_size[0] * scale)
            sw = int(page.orig_size[1] * scale)
            out_image = Image.fromarray(out_image[:sh, :sw])
            imageio = cStringIO.StringIO()
            out_image.save(imageio, 'png')
            imageio.seek(0)
            out.writestr(imslpid + '-' + str(pagenum) + '.png', imageio.read())
        except Exception, e:
            print e

        if i % 100 == 0:
            out.close()
            shutil.copy('trainset.zip', 'trainset-current.zip')
            out = zipfile.ZipFile('trainset.zip', 'a', zipfile.ZIP_DEFLATED, True)
            runs.flush()
        del page
        gc.collect()
        pagenum += 1
        i += 1
