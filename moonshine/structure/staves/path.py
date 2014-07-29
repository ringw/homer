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
    page.paths = page.paths.get().view(path_point).reshape(page.paths.shape[:2])
    return page.paths

def stable_paths(page):
    if not hasattr(page, 'paths'):
        paths(page)
    
    starting_points = [[] for i in xrange(page.paths.shape[1])]
    # Trace each endpoint back along its shortest path
    x = page.paths.shape[1] - 1
    path_ys = np.arange(page.paths.shape[0])
    while x > 0:
        path_ys = page.paths[path_ys, x]['prev']
        x -= 1
    for start_y, end_y, cost in zip(path_ys,
                                    xrange(page.paths.shape[0]),
                                    page.paths[:,-1]['cost']):
        starting_points[start_y].append((cost, end_y))
    return starting_points
