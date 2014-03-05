from PIL import Image
import tempfile
import os
import shutil
from StringIO import StringIO
import subprocess
import numpy as np
try:
  from .pdfimage import pdf_to_images
except ImportError:
  print "PDF import disabled; please install pylibtiff"
  pdf_to_images = None

IMAGE_MAX_SIZE = 4096
def image_array(data):
  if type(data) is str:
    im = Image.open(StringIO(data))
  else:
    im = data
  im = im.convert('1')
  if im.size[0] > IMAGE_MAX_SIZE and im.size[0] > im.size[1]:
    im = im.resize((IMAGE_MAX_SIZE, im.size[1]*IMAGE_MAX_SIZE/im.size[0]))
  elif im.size[1] > IMAGE_MAX_SIZE:
    im = im.resize((im.size[0]*IMAGE_MAX_SIZE/im.size[1], IMAGE_MAX_SIZE))
  im = im.convert('L')
  bytestring = im.tobytes()
  pixels = np.fromstring(bytestring, dtype=np.uint8)
  pixels = pixels.reshape((im.size[1], im.size[0]))
  # Heuristic to detect whether image needs inverting
  if (pixels != 0).sum() > im.size[0] * im.size[1] / 2:
    # Swap pixels so colored pixels are 1
    np.logical_not(pixels, output=pixels)
  else:
    pixels[pixels != 0] = 1
  return pixels

# Open image or multi-page PDF, return list of data (or PIL Images)
def read_pages(path):
  if isinstance(path, basestring):
    path = open(path, 'rb')
  images = []
  path.seek(0)
  if path.read(4) == '%PDF':
    if pdf_to_images is None:
      raise ValueError("pylibtiff required for PDF import")
    images = pdf_to_images(path)
  else:
    path.seek(0)
    images = [path.read()]
  return images
