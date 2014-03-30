import moonshine

import logging
logging.basicConfig(level=logging.INFO)

import sys
#args = moonshine.parser.parse_args()
page = moonshine.open(sys.argv[1])[0]
page.process()
page.show()
