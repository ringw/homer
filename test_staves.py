# Test staff/system recognition on solo piano sheet music.
# This assumes an even number of staves where systems are 2 staves each.
import moonshine
from moonshine import structure, opencl
import sys
import gc
import logging

good_pages = 0
total_pages = 0
for doc in sys.argv[1:]:
    logging.warn(doc)
    pages = moonshine.open(doc)
    for i, page in enumerate(pages):
        structure.process(page)
        # All staves should start at the same x coordinate, except possibly
        # at the start of a movement where 2 staves are indented
        # All staves should always end at the same x coordinate
        MAX_DIST = 2*page.staff_dist
        starts = page.staves[:, 0]
        start_maxdiff = max(starts) - min(starts) if len(starts) else 0
        if start_maxdiff > MAX_DIST:
            # Search for start of a movement
            min_start = min(starts)
            max_start = max(starts)
            if (((starts - min_start) >= MAX_DIST).sum() == 2
                and ((max_start - starts) < MAX_DIST).sum() == 2):
                start_maxdiff = 0
        ends = page.staves[:, 1]
        end_maxdiff = max(ends) - min(ends) if len(ends) else 0
        if start_maxdiff > MAX_DIST:
            print 'Staves start in different places at %s p.%d' % (doc, i)
        elif end_maxdiff > MAX_DIST:
            print 'Staves end in different places at %s p.%d' % (doc, i)
        elif len(page.staves) == 0:
            print 'No staves at %s p.%d' % (doc, i)
        elif len(page.staves) % 2 != 0:
            print 'Odd # staves at %s p.%d' % (doc, i)
        elif not all([s['stop'] - s['start'] == 1 for s in page.systems]):
            print 'System with != 2 staves at %s p.%d' % (doc, i)
        elif not all([len(s['barlines']) > 2 for s in page.systems]):
            print 'Only 2 barlines at %s p.%d' % (doc, i)
        else:
            good_pages += 1
        total_pages += 1
        opencl.q.finish()
    del pages
    gc.collect()

print 'accuracy %0.02d (%d/%d)' % (float(good_pages) * 100 / total_pages,
                                   good_pages, total_pages)
