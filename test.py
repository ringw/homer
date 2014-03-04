import moonshine

INPUT = 'samples/IMSLP73123.pdf'

score = moonshine.open(INPUT)
for page in score:
  page.process()
  page.show()
import pylab
pylab.show()
