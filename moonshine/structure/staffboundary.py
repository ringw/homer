from ..gpu import *
from .. import util
import numpy as np

prg = build_program(["boundary", "scaled_bitmap_to_int_array",
                     "taxicab_distance"])

def boundary_cost_kernel(dist, y0, ystep, y1, x0, xstep, x1):
    numy = int(y1 - y0) // int(ystep)
    numx = int(x1 - x0) // int(xstep)
    costs = cla.zeros(q, (numx, numy, numy), np.float32)
    prg.boundary_cost(q, (numx, numy, numy), (1, 1, 1),
                                  dist.data,
                                  np.int32(dist.shape[0]),
                                  np.int32(y0),
                                  np.int32(ystep),
                                  np.int32(numy),
                                  np.int32(x0),
                                  np.int32(xstep),
                                  np.int32(numx),
                                  costs.data).wait()
    return costs

def distance_transform_kernel(img, numiters=64):
    for i in xrange(numiters):
        e = prg.taxicab_distance_step(q, img.shape[::-1],
                                                          (16, 32),
                                                          img.data)
    e.wait()

DT_SCALE = 5.0
def distance_transform(page):
    dt = cla.zeros(q, (2048, 2048), np.int32)
    prg.scaled_bitmap_to_int_array(q, dt.shape[::-1], (16, 16),
                                   page.img.data,
                                   np.float32(DT_SCALE),
                                   np.int32(page.img.shape[1]),
                                   np.int32(page.img.shape[0]),
                                   np.int32(64), np.int32(0),
                                   dt.data).wait()
    distance_transform_kernel(dt, numiters=64)
    page.distance_transform = dt
    return dt

def shortest_path(edge_costs, start_y):
    ptr = np.empty((edge_costs.shape[0], edge_costs.shape[1]), int)
    path_length = np.empty_like(ptr, dtype=float)
    ptr[1, :] = start_y
    path_length[1, :] = edge_costs[1, start_y]
    for i in xrange(2, edge_costs.shape[0]):
        possible_lengths = edge_costs[i] + path_length[i-1, :, None]
        ptr[i] = np.argmin(possible_lengths, axis=0)
        path_length[i] = np.amin(possible_lengths, axis=0)
    rev_path = []
    y = start_y
    for i in xrange(edge_costs.shape[0] - 1, 0, -1):
        rev_path.append((i, y))
        y = ptr[i, y]
    rev_path.append((0, start_y))
    return np.array(list(reversed(rev_path)))

def boundary_cost(page, staff):
    if staff == 0:
        y0 = 0
    else:
        y0 = max(0, np.amax(page.staves[staff-1, 2:4]) + page.staff_dist*2)
    if staff == len(page.staves):
        y1 = page.orig_size[0]
    else:
        y1 = min(page.orig_size[0],
                 np.amin(page.staves[staff, 2:4]) - page.staff_dist*2)

    # Try to find a horizontal line that doesn't touch any dark pixels
    proj = page.img[y0:y1].get().sum(axis=1)
    slices, num_slices = util.label_1d(proj == 0)
    if slices.any():
        slice_size = np.bincount(slices)
        slice_num = np.argmax(slice_size[1:]) + 1
        slice_pixels, = np.where(slices == slice_num)
        slice_y = y0 + int(np.mean(slice_pixels))
        return np.array([[0, slice_y], [page.orig_size[1], slice_y]])
    y0 /= DT_SCALE
    y1 /= DT_SCALE
    xstep = ystep = page.staff_thick
    x0 = 0
    x1 = 2048
    edge_costs = boundary_cost_kernel(page.distance_transform,
                                      int(y0), ystep, int(y1),
                                      int(x0), xstep, int(x1)).get()
    if staff == 0:
        start_y = edge_costs.shape[1] - 1
    elif staff == len(page.staves):
        start_y = 0
    else:
        start_y = edge_costs.shape[1] // 2
    path = shortest_path(edge_costs, start_y)
    path[:, 0] = DT_SCALE * (x0 + xstep * path[:, 0])
    path[:, 1] = DT_SCALE * (y0 + ystep * path[:, 1])
    if path[-1, 0] < x1:
        path = np.concatenate((path, [[x1, DT_SCALE * (y0 + ystep * start_y)]]))
    return path

def boundaries(page):
    distance_transform(page)
    boundaries = []
    for i in xrange(len(page.staves) + 1):
        boundaries.append(boundary_cost(page, i))
    page.boundaries = boundaries
    return boundaries

def show_boundaries(page):
    import pylab as p
    for b in page.boundaries:
        p.plot(*(tuple(b.T) + ('m',)))
