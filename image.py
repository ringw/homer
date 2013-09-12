import Image
import tempfile
import os
import shutil
from StringIO import StringIO
import subprocess
import numpy as np
import page

def image_array(f):
  im = Image.open(f)
  im = im.convert('1')
  im = im.convert('L')
  bytestring = im.tostring()
  pixels = np.fromstring(bytestring, dtype=np.uint8)
  pixels = pixels.reshape((im.size[1], im.size[0]))
  # Swap pixels so colored pixels are 1
  np.logical_not(pixels, output=pixels)
  return (pixels, im.convert('RGB'))

# Open image or multi-page PDF, return list of pages
def read_pages(path):
  if isinstance(path, basestring):
    path = open(path)
  path.seek(0)
  if path.read(4) == '%PDF':
    path.seek(0)
    #outFile = tempfile.NamedTemporaryFile(suffix='.png')
    tmpDir = tempfile.mkdtemp(prefix='moonshineTemp')
    tmpFormat = os.path.join(tmpDir, "page%05d.png")
    ARGS = ['gs', '-dBATCH', '-dNOPAUSE', '-dSAFER',
            '-sDEVICE=pngmono', '-r300',
            '-sOutputFile=' + tmpFormat,
            path.name]
    gs = subprocess.Popen(ARGS)
    gs.wait()
    if gs.returncode != 0: return False
    pages = []
    for filename in sorted(os.listdir(tmpDir)):
      pages.append(page.Page(*image_array(os.path.join(tmpDir, filename))))
    shutil.rmtree(tmpDir)
    return pages
  else:
    path.seek(0)
    return [page.Page(*image_array(path))]
