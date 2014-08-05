from ...opencl import *

prg = build_program('staff_paths')

path_point = np.dtype([('cost', np.float32), ('prev', np.int32)])
def paths(page, num_workers=1024, scale=2.0):
    page.paths = cla.zeros(q, (int(page.img.shape[0]/scale),
                               int(page.img.shape[1]*8/scale), 2),
                              np.int32)
    prg.staff_paths(q, (num_workers,), (num_workers,),
                    page.img.data,
                    np.int32(page.img.shape[1]),
                    np.int32(page.img.shape[0]),
                    np.int32(page.staff_thick),
                    np.float32(2.0),
                    page.paths.data,
                    np.int32(page.paths.shape[1]),
                    np.int32(page.paths.shape[0]))
    return page.paths

def stable_paths(page):
    if not hasattr(page, 'paths'):
        paths(page)
    h = page.paths.shape[0]
    w = page.paths.shape[1]
    path_end = cla.zeros(q, (h,), np.int32)
    prg.find_stable_paths(q, (h,), (512,),
                          page.paths.data,
                          np.int32(w),
                          path_end.data).wait()
    path_end = path_end.get()
    stable_path_end = path_end[path_end >= 0]
    if not len(stable_path_end):
        return np.zeros((0, w), int)
    stable_path_end = cla.to_device(q, stable_path_end.astype(np.int32))
    path_list = cla.empty(q, (len(stable_path_end), w), np.int32)
    prg.extract_stable_paths(q, (len(stable_path_end),), (1,),
                             page.paths.data,
                             np.int32(w),
                             stable_path_end.data,
                             path_list.data).wait()
    return path_list

def stable_paths_py(page):
    if not hasattr(page, 'paths'):
        paths(page)
    
    starting_points = [[] for i in xrange(page.paths.shape[1])]
    # Trace each endpoint back along its shortest path
    x = page.paths.shape[1] - 1
    path_ys = np.arange(page.paths.shape[0])
    paths_py = page.paths.get().view(path_point).reshape(page.paths.shape[:2])
    while x > 0:
        path_ys = paths_py[path_ys, x]['prev']
        x -= 1
    for start_y, end_y, cost in zip(path_ys,
                                    xrange(paths_py.shape[0]),
                                    paths_py[:,-1]['cost']):
        starting_points[start_y].append((cost, end_y))
    stable_paths = []
    for from_y0 in starting_points:
        if not from_y0:
            continue
        cost_to_path = dict(from_y0)
        path_y = cost_to_path[min(cost_to_path)] # end y of min cost path
        path = []
        for x in xrange(paths_py.shape[1]-1, -1, -1):
            path.append(path_y)
            path_y = paths_py[path_y, x]['prev']
        stable_paths.append(list(reversed(path)))
    return stable_paths

def remove_paths(page, img, stable_paths):
    prg.remove_paths(q, img.shape[::-1], (16, 16),
                        img.data,
                        stable_paths.data,
                        np.int32(stable_paths.shape[1]),
                        np.int32(stable_paths.shape[0]),
                        np.int32(2)) # XXX
    return img
