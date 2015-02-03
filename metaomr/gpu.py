from reikna import cluda
from reikna.core import Type
api = cluda.ocl_api()
import pyopencl
thr = api.Thread(pyopencl.create_some_context())
import numpy as np

import os.path
SRC_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../kernels')
def build_program(filenames):
    if type(filenames) is not list:
        filenames = [filenames]
    filenames = ['preamble'] + filenames
    paths = [os.path.join(SRC_ROOT, f + ".cl") for f in filenames]
    src = "".join([open(path, "rb").read() for path in paths])
    return thr.compile(src)
