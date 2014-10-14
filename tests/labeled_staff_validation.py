# Comparative staff detection and removal accuracy
import env
import moonshine
from moonshine.staves import validation, hough, path, dummy
from moonshine.staves.gamera_musicstaves import *
from moonshine import orientation, staffsize
from moonshine import page as page_mod

import gamera.plugins.numpy_io
from gamera.toolkits.musicstaves.plugins import staffdeformation

TESTSET = './musicstaves-testset-modern'

methods = dict(hough=hough.FilteredHoughStaves,
               #path=path.StablePathStaves,
               #linetracking=MusicStaves_linetracking,
               #carter=MusicStaves_rl_carter,
               fujinaga=MusicStaves_rl_fujinaga,
               #roach_tatem=MusicStaves_rl_roach_tatem,
               #gamera_simple=MusicStaves_rl_simple,
               skeleton=MusicStaves_skeleton,
               dalitz=StaffFinder_dalitz,
               )#miyao=StaffFinder_miyao,
               #projections=StaffFinder_projections)
musicstaves_methods = ['fujinaga', 'skeleton']

def kanungo(eta, a0, a, b0, b, k=2, seed=1):
    return lambda x,y: staffdeformation.degrade_kanungo_parallel.__call__(
                            x, y, eta, a0, a, b0, b, k, seed)
deformations = dict([('orig', lambda x,y: [x,y]),
                     ('k0.001-.1-.5', kanungo(0.001, 0.1, 0.5, 0.1, 0.5)),
                     ('k0.001-1-.9-.1-.5', kanungo(0.001, 1, 1, 0.1, 0.5)),
                     ('k0.05-1-.9-.1-.5', kanungo(0.05, 1, 1, 0.1, 0.5)),
                     ('k0.05-1-.5-.1-.5', kanungo(0.05, 1, .5, 0.1, 0.5)),
                     ('k-.5-.1-.1-.5', kanungo(0, 0.5, 0.1, 0.1, 0.5)),
                     ('k0.02-0.1-0.1-0.5-0.1', kanungo(0.02, 0.1, 0.1, 0.5, 0.1)),
                     ('k-sp-0.05', kanungo(0.05, 0, 0, 0, 0)),
                     ('curvature-0.02-0.1', lambda x,y: staffdeformation.curvature.__call__(x,y, 0.02,0.1)),
                     ('curvature-0.01-0.05', lambda x,y: staffdeformation.curvature.__call__(x,y, 0.02,0.1)),
                     ('curvature-0.005-0.2', lambda x,y: staffdeformation.curvature.__call__(x,y, 0.02,0.1))])

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
    pagedata = pandas.DataFrame(columns='staff_sens staff_spec'.split())
    for i, filename in enumerate(sorted(glob.glob(TESTSET + '/*-nostaff.png'))):
        gc.collect()
        fileid = os.path.basename(filename).split('-')[0]
        orig_file = re.sub('-nostaff', '', filename)
        page, = moonshine.open(orig_file)
        nostaff, = moonshine.open(filename)

        staffsize.staffsize(page)
        if type(page.staff_dist) is tuple or page.staff_dist is None:
            continue
        orientation.rotate(page)
        staffsize.staffsize(page)
        orientation.rotate(nostaff, page.orientation)

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
            page_runs = staffsize.light_runs(page_)[page_.staff_space]

            validator = validation.StaffValidation(page_)
            dummy_ = dummy.DummyStaves(page_, nostaff_)
            if deformation == 'orig':
                baselineStaves = dummy_()
            else:
                dummy_.staves = baselineStaves
            scores = validator.score_staves(method=dummy_)
            scores.index = ('dummy-%s-%s-' % (fileid,deformation)) + scores.index
            result = result.append(scores)
            def handler(signum, frame):
                raise Exception('...timeout')
            for method in methods:
                print '%s-%s-%s' % (method,fileid,deformation)
                try:
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(30)
                    staves = methods[method](page_)
                    scores = validator.score_staves(method=staves)
                    scores.index = ('%s-%s-%s-' % (method, fileid,deformation)) + scores.index
                    sens, spec = staves.score(baselineStaves)
                    pagedata.loc[('%s-%s-%s' % (method, fileid,deformation))] = [sens, spec]
                    signal.alarm(0)
                except Exception, e:
                    signal.alarm(0)
                    print e
                    scores = pandas.DataFrame(dict(runs=page_runs,
                                                   removed=0),
                                              index=['%s-%s-%s-page' % (method,fileid,deformation)])
                result = result.append(scores)
            for method in musicstaves_methods:
                print '%s-native-%s-%s' % (method,fileid,deformation)
                try:
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(30)
                    staves = methods[method](page_, staff_removal='gamera')
                    scores = validator.score_staves(method=staves)
                    scores.index = ('%s-native-%s-%s-' % (method, fileid,deformation)) + scores.index
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
        result.to_csv(open(output, 'wb'))
        pagedata.to_csv(open('pagedata-'+output, 'wb'))
finally:
    shutil.rmtree(tmpdir)
