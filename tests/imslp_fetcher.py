import urllib2
from time import sleep
import random
import os
from tempfile import mkdtemp
import glob
import subprocess
import shutil
import re
import json
import sys
from datetime import datetime

def imslp_bot():
    opener = urllib2.build_opener()
    opener.addheaders.append(('Cookie', 'imslpdisclaimeraccepted=yes'))
    count = 0
    while True:
        if count:
            sleep(30)
        count += 1
        num = random.randint(1, 320000)
        f = opener.open('http://imslp.org/wiki/Special:ImagefromIndex/%05d' % num)
        if f.headers['Content-Type'] != 'application/pdf':
            continue
        if 'Content-Length' not in f.headers:
            print 'no Content-Length!'
            continue
        cl = int(f.headers['Content-Length'])
        if 50000 <= cl and cl <= 2000000:
            yield f

def imslp_fetcher():
    if not os.path.exists('/tmp/IMSLP'):
        os.mkdir('/tmp/IMSLP')
    bot = imslp_bot()
    for req in bot:
        data = req.read()
        path = os.path.join('/tmp/IMSLP', os.path.basename(req.geturl()))
        open(path, 'wb').write(data)
        tmpdir = mkdtemp()
        subprocess.call(['/usr/bin/pdfimages', path, os.path.join(tmpdir,'page')])
        name, pages = os.path.basename(req.geturl()), sorted(glob.glob(os.path.join(tmpdir,'page-*')))
        if all(['.pbm' in p for p in pages]):
            yield name, pages
        shutil.rmtree(tmpdir)
