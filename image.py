import Image
import tempfile
from StringIO import StringIO
import subprocess
import numpy as np
import page

def image_array(f):
  im = Image.open(f)
  im = im.convert('1')
  # White: 0, Black: 1
  arr = 1 - np.array(list(im.getdata()), dtype=np.int)/255
  return (arr.reshape(im.size[1], im.size[0]), im.convert('RGB'))

# Open image or multi-page PDF, return list of pages
def read_pages(path):
  if isinstance(path, basestring):
    path = open(path)
  path.seek(0)
  if path.read(4) == '%PDF':
    path.seek(0)
    outFile = tempfile.NamedTemporaryFile(suffix='.png')
    ARGS = ['gs', '-dBATCH', '-dNOPAUSE', '-dSAFER',
            '-sDEVICE=pngmono', '-r300',
            '-sOutputFile=' + outFile.name,
            path.name]
    gs = subprocess.Popen(ARGS)
    gs.wait()
    if gs.returncode != 0: return False
    outFile.seek(0)
    return [page.Page(*image_array(outFile))]
  path.seek(0)
  return [page.Page(*image_array(path))]
