import Image
import tempfile
import os
import shutil
from StringIO import StringIO
import subprocess
import numpy as np
import page

IMAGE_MAX_SIZE = 4096
def image_array(data):
  im = Image.open(StringIO(data))
  im = im.convert('1')
  if im.size[0] > IMAGE_MAX_SIZE and im.size[0] > im.size[1]:
    im = im.resize((IMAGE_MAX_SIZE, im.size[1]*IMAGE_MAX_SIZE/im.size[0]))
  elif im.size[1] > IMAGE_MAX_SIZE:
    im = im.resize((im.size[0]*IMAGE_MAX_SIZE/im.size[1], IMAGE_MAX_SIZE))
  im = im.convert('L')
  bytestring = im.tostring()
  pixels = np.fromstring(bytestring, dtype=np.uint8)
  pixels = pixels.reshape((im.size[1], im.size[0]))
  # Swap pixels so colored pixels are 1
  np.logical_not(pixels, output=pixels)
  return (pixels, im.convert('RGB'))

def pdf_to_pngs(path):
  tmpDir = tempfile.mkdtemp(prefix='moonshineTemp')
  tmpFormat = os.path.join(tmpDir, "page%05d.png")
  ARGS = ['gs', '-dBATCH', '-dNOPAUSE', '-dSAFER',
          '-sDEVICE=pngmono', '-r300',
          '-sOutputFile=' + tmpFormat,
          path.name]
  gs = subprocess.Popen(ARGS)
  gs.wait()
  if gs.returncode != 0: return False
  images = []
  for filename in sorted(os.listdir(tmpDir)):
    images.append(open(os.path.join(tmpDir, filename)).read())
  shutil.rmtree(tmpDir)
  return images
# Open image or multi-page PDF, return list of pages
def read_pages(path):
  if isinstance(path, basestring):
    path = open(path)
  images = []
  path.seek(0)
  if path.read(4) == '%PDF':
    images = pdf_to_pngs(path)
  else:
    path.seek(0)
    images = [path.read()]
  return map(page.Page, images)
