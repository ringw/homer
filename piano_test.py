# We can test scores for piano solo because we know what to expect
# (an even number of staves and staff systems with 2 staves each.)
# This test can be used with any scores from IMSLP or other sources.

import moonshine
import moonshine.opencl
import gc
import sys

PASS = 0
TOTAL = 0
for path in sys.argv[1:]:
    score = moonshine.open(path)
    for i,page in enumerate(score):
        page.process()
        TOTAL += 1
        if len(page.staves) < 10 or len(page.staves) > 16:
            print "Wrong number of staves on", path, "page", i
        elif len(page.staves) % 2 != 0:
            print "Odd number of staves on", path, "page", i
        else:
            for start, end, barlines in page.barlines:
                if end - start != 2:
                    print "Wrong system size", path, "page", i
                    break
            else:
                PASS += 1
    moonshine.opencl.q.finish()
    del score
    del page
    gc.collect()

print PASS, "pages correct out of", TOTAL, str(float(PASS) / TOTAL * 100) + "%"
