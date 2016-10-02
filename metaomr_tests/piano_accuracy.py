# Test staff/system recognition on solo piano sheet music.
# This assumes an even number of staves where systems are 2 staves each.
import env
import metaomr
from metaomr import structure
import sys
import gc
import logging

good_pages = 0
total_pages = 0
for doc in sys.argv[1:]:
    logging.warn(doc)
    pages = metaomr.open(doc)
    for i, page in enumerate(pages):
        structure.staffsize.staffsize(page)
        if type(page.staff_dist) is tuple:
            print 'Multiple staff sizes at %s p.%d' % (doc, i)
        else:
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
    del pages
    gc.collect()

print 'accuracy %0.02f%% (%d/%d)' % (float(good_pages) * 100 / total_pages,
                                     good_pages, total_pages)
