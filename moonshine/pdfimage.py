try:
  from PyPDF2 import PdfFileReader
except ImportError:
  from pyPdf import PdfFileReader
from PIL import Image
from libtiff import TIFF, libtiff_ctypes
from tempfile import NamedTemporaryFile
from StringIO import StringIO

def pdf_page_extract_image(page):
  if not '/Resources' in page:
    return None
  res = page['/Resources']
  if not '/XObject' in res:
    return None
  xobject = res['/XObject']
  if not '/ProcSet' in res:
    return None
  has_image = False
  for proc in res['/ProcSet']:
    if len(proc) >= 6 and proc[:2] == '/Image':
      has_image = True
      break
  if len(xobject) > 1:
    return None
  im = xobject[xobject.keys()[0]]
  if im['/Filter'] == '/CCITTFaxDecode':
    tiff_file = NamedTemporaryFile(suffix='.tiff')
    del im['/Filter']
    data = im.getData()
    t = libtiff_ctypes.libtiff.TIFFOpen(tiff_file.name, 'w')
    t.SetField(libtiff_ctypes.TIFFTAG_IMAGEWIDTH, im['/Width'])
    t.SetField(libtiff_ctypes.TIFFTAG_IMAGELENGTH, im['/Height'])
    t.SetField(libtiff_ctypes.TIFFTAG_BITSPERSAMPLE, im['/BitsPerComponent'])
    t.SetField(libtiff_ctypes.TIFFTAG_COMPRESSION, 4) # ccitt4
    t.SetField(libtiff_ctypes.TIFFTAG_SAMPLESPERPIXEL, 1) # XXX
    t.WriteRawStrip(0, data, len(data))
    t.WriteDirectory()
    tiff_file.seek(0)
    return tiff_file.read()
  else:
    try:
      bits = im.getData()
      assert im['/Width'] % 8 == 0
      assert im['/Width'] * im['/Height'] / 8 == len(bits)
      #img = Image.frombytes('1', (im['/Width'], im['/Height']), bits)
      #buf = StringIO()
      #img.save(buf, format='PNG')
      #return buf.getvalue()
      return Image.frombytes('1', (im['/Width'], im['/Height']), bits)
    except:
      return None

def pdf_to_images(pdf):
  if type(pdf) is str:
    pdf = open(pdf, 'rb')
  pdf = PdfFileReader(pdf)
  tiffs = []
  for p in xrange(pdf.getNumPages()):
    data = pdf_page_extract_image(pdf.getPage(p))
    if data is not None:
      tiffs.append(data)
  return tiffs
