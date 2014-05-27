import moonshine

import logging
logging.basicConfig(level=logging.INFO)

args = moonshine.parser.parse_args()
score = moonshine.open(args.path)
if args.page is None:
    for page in score:
        page.process()
        if args.show:
            page.show()
else:
    page = score[args.page]
    page.process()
    if args.show:
        page.show()
if args.show:
    import pylab as p
    p.show()
