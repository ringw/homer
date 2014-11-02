# Connected component labelling using Union-Find
from .gpu import *
import numpy as np

prg = build_program('components')
int2 = np.dtype('i4,i4')

def get_components(classes_img):
    # Pad classes_img as necessary
    if any(s % 8 != 0 for s in classes_img.shape):
        classes_img = classes_img.get()
        pad_h = -(-classes_img.shape[0] & -8)
        pad_w = -(-classes_img.shape[1] & -8)
        classes_pad = np.zeros((pad_h, pad_w), np.uint8)
        classes_pad[:classes_img.shape[0], :classes_img.shape[1]] = classes_img
        classes_img = thr.to_device(classes_pad)

    pixel_tree = thr.empty_like(Type(np.int32, classes_img.shape))
    pixel_tree.fill(0)
    prg.init_component_tree(classes_img.data, pixel_tree.data,
                            global_size=pixel_tree.shape[::-1],
                            local_size=(8,8))
    prg.build_component_tree(classes_img.data, pixel_tree.data,
                             global_size=pixel_tree.shape[::-1],
                             local_size=(8,8))
    num_components = thr.empty_like(Type(np.int32, (1,)))
    num_components.fill(0)
    prg.count_components(pixel_tree.data, num_components.data,
                         global_size=pixel_tree.shape[::-1],
                         local_size=(8,8))
    num_components = int(num_components.get()[0])
    bounds = thr.empty_like(Type(np.int32, (num_components, 4)))
    prg.init_component_bounds(bounds.data,
                              np.array(classes_img.shape[::-1],
                                       np.int32).view(int2)[0],
                              global_size=(num_components,),
                              local_size=(1,))
    classes = thr.empty_like(Type(np.uint8, num_components))
    classes.fill(0)
    sums = thr.empty_like(Type(np.int32, num_components))
    sums.fill(0)
    prg.component_info(classes_img.data, pixel_tree.data,
                       classes.data, bounds.data, sums.data,
                       global_size=pixel_tree.shape[::-1],
                       local_size=(8, 8))
    return classes, bounds, sums
