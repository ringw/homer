import moonshine
import argparse
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--page", type=int)
parser.add_argument("-s", "--show", dest="show", action="store_true")
parser.add_argument("-S", "--no-show", dest="show", action="store_false")
parser.add_argument("-i", "--interactive", dest="int", action="store_true")
parser.add_argument("-o", "--output", type=str)
parser.set_defaults(show=True)
parser.add_argument("path", type=str, help="path to scanned music")

args = parser.parse_args()
score = moonshine.open(args.path)
if args.page is None:
    if args.output:
        from matplotlib.backends.backend_pdf import PdfPages
        import pylab as p
        with PdfPages(args.output) as pdf:
            for page in score:
                page.process()
                p.figure(figsize=page.orig_size)
                ax = p.Axes(p.gcf(),[0,0,1,1],yticks=[],xticks=[],frame_on=False)
                p.gcf().delaxes(p.gca())
                p.gcf().add_axes(ax)
                page.show()
                pdf.savefig()
                p.close()
    else:
        for page in score:
            page.process()
            if args.show:
                page.show()
else:
    page = score[args.page]
    page.process()
    if args.show:
        page.show()
if args.show and not args.output:
    import pylab as p
    if args.int:
        p.ion()
    p.show()
