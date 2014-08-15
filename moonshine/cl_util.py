# General utilities for OpenCL arrays
from .gpu import *
from reikna.core import Type
from reikna.cluda import Snippet
from reikna.algorithms.reduce import Reduce, Predicate

prg = build_program(["maximum_filter", "taxicab_distance"])
def maximum_filter_kernel(img):
    maximum = thr.empty_like(img)
    maximum[:] = 0
    prg.maximum_filter(img.data, maximum.data,
                       global_size=map(int, img.shape[::-1]),
                       local_size=(1, 1))
    return maximum

max_snippet = Snippet.create(lambda a, b: """
    return ((${a}) > (${b})) ? (${a}) : (${b});
""")
def max_kernel(arr):
    max_func = Reduce(arr, Predicate(max_snippet, np.array([-10.0**9],np.float32)[0])).compile(thr)
    out = thr.empty_like(Type(np.float32))
    max_func(out, arr)
    return out.get().item()
