import moonshine
import argparse
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--page", type=int)
parser.add_argument("-s", "--show", dest="show", action="store_true")
parser.add_argument("-S", "--no-show", dest="show", action="store_false")
parser.add_argument("-i", "--interactive", dest="int", action="store_true")
parser.set_defaults(show=True)
parser.add_argument("path", type=str, help="path to scanned music")

args = parser.parse_args()
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
    if args.int:
        p.ion()
    p.show()
