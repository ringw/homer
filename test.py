import moonshine

INPUT = 'samples/sonata.png'

score = moonshine.open(INPUT)[0]
score.process()
score.show()
import pylab
pylab.show()
