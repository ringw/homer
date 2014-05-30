# Connected component labelling using Union-Find
from .opencl import *
import numpy as np

prg = build_program('components')
int2 = cl.tools.get_or_register_dtype('int2')
prg.init_component_bounds.set_scalar_arg_dtypes([None, int2])

def get_components(classes_img):
    pixel_tree = cla.zeros(q, classes_img.shape, np.int32)
    prg.init_component_tree(q, pixel_tree.shape[::-1], (8,8),
                            classes_img.data, pixel_tree.data)
    prg.build_component_tree(q, pixel_tree.shape[::-1], (8,8),
                             classes_img.data, pixel_tree.data)
    num_components = cla.zeros(q, (1,), np.int32)
    prg.count_components(q, pixel_tree.shape[::-1], (8,8),
                         pixel_tree.data, num_components.data)
    num_components = int(num_components.get()[0])
    bounds = cla.empty(q, (num_components, 4), np.int32)
    prg.init_component_bounds(q, (num_components,), (1,),
                              bounds.data,
                              np.array(classes_img.shape[::-1],
                                       np.int32).view(int2)[0]).wait()
    classes = cla.zeros(q, num_components, np.uint8)
    sums = cla.zeros(q, num_components, np.int32)
    prg.component_info(q, pixel_tree.shape[::-1], (8, 8),
                       classes_img.data, pixel_tree.data,
                       classes.data, bounds.data, sums.data)
    return classes, bounds, sums
