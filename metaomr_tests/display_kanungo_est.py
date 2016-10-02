import env
import metaomr
import metaomr.kanungo as k
import sys

page, = metaomr.open(sys.argv[1])
page.preprocess()
print k.est_parameters(page)
