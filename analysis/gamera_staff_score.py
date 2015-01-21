# Comparative staff detection and removal accuracy
import env
import moonshine
from moonshine.staves import validation, hough, path, dummy
from moonshine.staves.gamera_musicstaves import *
from moonshine import orientation, staffsize
from moonshine import page as page_mod
import datetime
import gzip
import numpy as np

import gamera.plugins.numpy_io
from gamera.toolkits.musicstaves.plugins import staffdeformation

TESTSET = './musicstaves-testset-modern'

def hough_staves(thetarange, ntheta):
    def get_hough(page):
        h = hough.FilteredHoughStaves(page)
        h.thetas = np.linspace(-thetarange, thetarange, ntheta)
        return h
    return get_hough

methods = dict(#hough_pi250_201=hough_staves(np.pi/250, 201),
               #hough_1deg_201=hough_staves(np.pi/180, 201),
               #hough_1deg_51=hough_staves(np.pi/180, 51),
               #hough_hdeg_21=hough_staves(np.pi/360, 21),
               hough=hough_staves(np.pi/250,201),
               #path=path.StablePathStaves,
               #linetracking=MusicStaves_linetracking,
               #carter=MusicStaves_rl_carter,
               fujinaga=MusicStaves_rl_fujinaga,
               #roach_tatem=MusicStaves_rl_roach_tatem,
               #gamera_simple=MusicStaves_rl_simple,
               skeleton=MusicStaves_skeleton,
               dalitz=StaffFinder_dalitz,
               miyao=StaffFinder_miyao,
               projections=StaffFinder_projections)
musicstaves_methods = ['fujinaga', 'skeleton']

def kanungo(eta, a0, a, b0, b, k=2, seed=42):
    return lambda x,y: staffdeformation.degrade_kanungo_parallel.__call__(
                            x, y, eta, a0, a, b0, b, k, seed)
deformations = dict([('k0.001-.1-.5', kanungo(0.001, 0.1, 0.5, 0.1, 0.5)),
                     ('k0.001-1-.9-.1-.5', kanungo(0.001, 1, 1, 0.1, 0.5)),
                     ('k0.05-1-.9-.1-.5', kanungo(0.05, 1, 1, 0.1, 0.5)),
                     ('k0.05-1-.5-.1-.5', kanungo(0.05, 1, .5, 0.1, 0.5)),
                     ('k-.5-.1-.1-.5', kanungo(0, 0.5, 0.1, 0.1, 0.5)),
                     ('k0.02-0.1-0.1-0.5-0.1', kanungo(0.02, 0.1, 0.1, 0.5, 0.1)),
                     ('k-sp-0.05', kanungo(0.05, 0, 0, 0, 0))])
for k in [0.01,0.005,0.001,0.0005,0.0001]:
    for l in [0.2,0.5,1,1.5,2]:
        deformations['curv%.3f-%.2f' % (k,l)] = lambda x,y: staffdeformation.curvature.__call__(x,y,k,l)
for p in [0.0001,0.001,0.01]:
    for n in [2,5,10]:
        for k in [1,2]:
            deformations['sp%.3f-%d-%d' % (p,n,k)] = lambda x,y: staffdeformation.white_speckles_parallel.__call__(x,y,p,n,k,random_seed=42)
for t in [-.1,.1,-.5,.5,-1,1,2,2]:
    deformations['rot%f' % t] = lambda x,y: staffdeformation.rotation.__call__(x,y,t)

import gc
import glob
import os
import pandas
import re
import shutil
import signal
import sys
import subprocess
import tempfile
tmpdir = tempfile.mkdtemp()

try:
    output = sys.argv[1]

    result = pandas.DataFrame()
    pagedata = pandas.DataFrame(columns='staff_sens staff_spec time'.split())
    for i, filename in enumerate(sorted(glob.glob(TESTSET + '/*-nostaff.png'))):
        gc.collect()
        fileid = os.path.basename(filename).split('-')[0]
        orig_file = re.sub('-nostaff', '', filename)
        page, = moonshine.open(orig_file)
        nostaff, = moonshine.open(filename)

        staffsize.staffsize(page)
        if type(page.staff_dist) is tuple or page.staff_dist is None:
            continue
        #orientation.rotate(page)
        #staffsize.staffsize(page)
        #orientation.rotate(nostaff, page.orientation)

        page_gamera = page.byteimg[:page.orig_size[0], :page.orig_size[1]]
        nostaff_gamera = nostaff.byteimg[:page.orig_size[0], :page.orig_size[1]]
        page_gamera = gamera.plugins.numpy_io.from_numpy(
                            page_gamera.astype(np.uint16))
        nostaff_gamera = gamera.plugins.numpy_io.from_numpy(
                            nostaff_gamera.astype(np.uint16))

        baselineStaves = None
        for deformation, func in [('orig', lambda x,y: [x,y])] + list(deformations.iteritems()):
            deformed = func(page_gamera, nostaff_gamera)
            page_d, nostaff_d = deformed[:2]
            page_np = gamera.plugins.numpy_io.to_numpy.__call__(page_d)
            if False and deformation != 'orig':
                import pylab
                pylab.imshow(page_np)
                pylab.show()
            nostaff_np = gamera.plugins.numpy_io.to_numpy.__call__(nostaff_d)
            page_ = page_mod.Page(page_np)
            nostaff_ = page_mod.Page(nostaff_np)
            page_.staff_dist = page.staff_dist
            page_.staff_thick = page.staff_thick
            page_.staff_space = page.staff_space
            page_runs = staffsize.staff_dist_hist(page_)[page_.staff_dist]

            validator = validation.StaffValidation(page_)
            dummy_ = dummy.LabeledStaffRemoval(page_, nostaff_)
            if deformation == 'orig':
                baselineStaves = dummy_()
            else:
                dummy_.staves = baselineStaves
            pagename = fileid + '-' + deformation

            page_pil = Image.fromarray((~page_np.astype(bool)).astype(np.uint8)*255)
            page_pil.save('musicstaves-testset-deformations/' + pagename + '.png')

            scores = validator.score_staves(method=dummy_)
            scores.index = 'dummy-' + pagename + scores.index.to_series()
            result = result.append(scores)
            def handler(signum, frame):
                raise Exception('...timeout')
            for method in methods:
                print method + '-' + pagename
                toc = None
                try:
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(30)
                    staves = methods[method](page_)
                    tic = datetime.datetime.now()
                    staves()
                    toc = datetime.datetime.now()
                    time = int((toc - tic).total_seconds() * 1000)
                    scores = validator.score_staves(method=staves)
                    scores.index = method + '-' + pagename + scores.index.to_series()
                    sens, spec = staves.score(baselineStaves)
                    pagedata.loc[method + '-' + pagename] = [sens, spec, time]
                    signal.alarm(0)
                except Exception, e:
                    signal.alarm(0)
                    print e
                    scores = pandas.DataFrame(dict(runs=page_runs,
                                                   removed=0),
                                              index=['%s-%s-page' % (method,pagename)])
                    pagedata.loc[method + '-' + pagename] = [0, 0, 30*1000 if toc is None else -1]
                result = result.append(scores)
            for method in musicstaves_methods:
                print '%s-native-%s-%s' % (method,fileid,deformation)
                try:
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(30)
                    staves = methods[method](page_, staff_removal='gamera')
                    scores = validator.score_staves(method=staves)
                    scores.index = ('%s-native-%s-%s-' % (method, fileid,deformation)) + scores.index.to_series()
                    signal.alarm(0)
                except Exception, e:
                    print e
                    scores = pandas.DataFrame(dict(runs=page_runs,
                                                   removed=0),
                                              index=['%s-%s-%s-page' % (method,fileid,deformation)])
                finally:
                    signal.alarm(0)
                result = result.append(scores)

    if len(result):
        result.to_csv(gzip.open(output, 'wb'))
        pagedata.to_csv(gzip.open('pagedata-'+output, 'wb'))
finally:
    shutil.rmtree(tmpdir)
