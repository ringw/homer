import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import logging

cx = cl.create_some_context()
q = cl.CommandQueue(cx, properties=cl.command_queue_properties.PROFILING_ENABLE)

import os.path
SRC_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../opencl')
def build_program(filenames):
    if type(filenames) is not list:
        filenames = [filenames]
    paths = [os.path.join(SRC_ROOT, f + ".cl") for f in filenames]
    src = "".join([open(path, "rb").read() for path in paths])
    return cl.Program(cx, src).build()
