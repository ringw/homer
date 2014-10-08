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

def kanungo(eta, a0, a, b0, b, k=2, seed=42):
    return lambda x,y: staffdeformation.degrade_kanungo_parallel.__call__(
                            x, y, eta, a0, a, b0, b, k, seed)
deformations = dict([('k0.01-0.1-0.2', kanungo(0.01, 0.1, 0.2, 0.1, 0.2))])

import gc
import glob
import gzip
import os
import pandas
import re
import shutil
import sys
import subprocess
import tempfile
tmpdir = tempfile.mkdtemp()

orientations = dict()
try:
    output = sys.argv[1]

    result = pandas.DataFrame()
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
        orientations[fileid] = float(page.orientation)

        page_gamera = page.byteimg[:page.orig_size[0], :page.orig_size[1]]
        nostaff_gamera = nostaff.byteimg[:page.orig_size[0], :page.orig_size[1]]
        page_gamera = gamera.plugins.numpy_io.from_numpy(
                            page_gamera.astype(np.uint16))
        nostaff_gamera = gamera.plugins.numpy_io.from_numpy(
                            nostaff_gamera.astype(np.uint16))

        validator = validation.StaffValidation(page)
        dummy_ = dummy.DummyStaves(page, nostaff)
        scores = validator.score_staves(method=dummy_)
        scores.index = ('dummy-%s-' % fileid) + scores.index
        result = result.append(scores)
        for method in methods:
            staves = methods[method](page)
            scores = validator.score_staves(method=staves)
            scores.index = ('%s-%s-' % (method, fileid)) + scores.index
            result = result.append(scores)
        gamera_image = None
        for method in musicstaves_methods:
            staves = methods[method](page, staff_removal='gamera')
            scores = validator.score_staves(method=staves)
            scores.index = ('%s-native-%s-' % (method, fileid)) + scores.index
            result = result.append(scores)
            if gamera_image is None:
                gamera_image = staves.gamera_image

        for deformation in deformations:
            func = deformations[deformation]
            deformed = func(page_gamera, nostaff_gamera)
            page_d, nostaff_d = deformed[:2]
            page_np = gamera.plugins.numpy_io.to_numpy.__call__(page_d)
            nostaff_np = gamera.plugins.numpy_io.to_numpy.__call__(nostaff_d)
            page_ = page_mod.Page(page_np)
            nostaff_ = page_mod.Page(nostaff_np)
            page_.staff_dist = page.staff_dist
            page_.staff_thick = page.staff_thick
            page_.staff_space = page.staff_space

            validator = validation.StaffValidation(page_)
            dummy_ = dummy.DummyStaves(page_, nostaff_)
            scores = validator.score_staves(method=dummy_)
            scores.index = ('dummy-%s-%s-' % (fileid,deformation)) + scores.index
            result = result.append(scores)
            for method in methods:
                staves = methods[method](page_)
                scores = validator.score_staves(method=staves)
                scores.index = ('%s-%s-%s-' % (method, fileid,deformation)) + scores.index
                result = result.append(scores)
            for method in musicstaves_methods:
                staves = methods[method](page_, staff_removal='gamera')
                scores = validator.score_staves(method=staves)
                scores.index = ('%s-native-%s-%s-' % (method, fileid, deformation)) + scores.index
                result = result.append(scores)

        break

    if len(result):
        result.to_csv(gzip.open(output, 'wb'))
finally:
    shutil.rmtree(tmpdir)
