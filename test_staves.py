# Test staff/system recognition on solo piano sheet music.
# This assumes an even number of staves where systems are 2 staves each.
import moonshine
from moonshine import structure, opencl
import sys
import gc

good_pages = 0
total_pages = 0
for doc in sys.argv[1:]:
    pages = moonshine.open(doc)
    for i, page in enumerate(pages):
        structure.process(page)
        if len(page.staves) % 2 != 0:
            print 'Odd # staves at %s p.%d' % (doc, i)
        elif not all([s['stop'] - s['start'] == 1 for s in page.systems]):
            print 'System with != 2 staves at %s p.%d' % (doc, i)
        else:
            good_pages += 1
        total_pages += 1
    del pages
    opencl.q.finish()
    gc.collect()

print 'accuracy %0.02d (%d/%d)' % (float(good_pages) * 100 / total_pages,
                                   good_pages, total_pages)
