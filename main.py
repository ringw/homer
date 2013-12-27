import moonshine
from moonshine import debug

args = moonshine.parser.parse_args()
if args.debug:
  debug.DEBUG_MODULES = args.debug.split(',')
moonshine.moonshine(args.path, show=args.show)
