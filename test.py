import metaomr

INPUT = 'samples/IMSLP73123.pdf'

score = metaomr.open(INPUT)
for page in score:
  page.process()
  page.show()
import pylab
pylab.show()
