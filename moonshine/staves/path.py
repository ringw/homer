from ..gpu import *
from .. import bitimage
from .base import BaseStaves
import logging

prg = build_program('staff_paths')

path_point = np.dtype([('cost', np.float32), ('prev', np.int32)])
class StablePathStaves(BaseStaves):
    scaled_img = None
    weights = None
    def get_weights(self, img=None, num_workers=512, scale=2.0):
        if img is None:
            img = self.page.img
        self.scaled_img = simg = bitimage.scale_image_gray(img, scale)
        self.weights = thr.empty_like(Type(np.int32, simg.shape + (2,)))
        self.weights.fill(0)
        prg.staff_paths(simg,
                        np.int32(simg.shape[1]),
                        np.int32(simg.shape[0]),
                        self.weights,
                        global_size=(num_workers,), local_size=(num_workers,))
        return self.weights

    def get_stable_paths(self, img=None):
        if img is None:
            img = self.page.img
        self.get_weights(img)
        h = self.weights.shape[0]
        w = self.weights.shape[1]
        path_end = thr.empty_like(Type(np.int32, h))
        prg.find_stable_paths(self.weights,
                              np.int32(w),
                              path_end,
                              global_size=(h,))
        path_end = path_end.get()
        stable_path_end = path_end[path_end >= 0]
        if not len(stable_path_end):
            return np.zeros((0, w), int)
        stable_path_end = thr.to_device(stable_path_end.astype(np.int32))
        path_list = thr.empty_like(Type(np.int32, (len(stable_path_end), w)))
        prg.extract_stable_paths(self.weights,
                                 np.int32(w),
                                 stable_path_end,
                                 path_list,
                                 global_size=(len(stable_path_end),),
                                 local_size=(1,))
        return path_list

    def get_stable_paths_py(self, img=None):
        if img is None:
            img = self.page.img
        self.get_weights(img)
        starting_points = [[] for i in xrange(self.weights.shape[1])]
        # Trace each endpoint back along its shortest path
        x = self.weights.shape[1] - 1
        path_ys = np.arange(self.weights.shape[0])
        paths_py = self.weights.get().view(path_point).reshape(self.weights.shape[:2])
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

    def validate_and_remove_paths(self, img, stable_paths):
        pixel_sum = thr.empty_like(Type(np.int32, len(stable_paths)))
        pixel_sum.fill(0)
        prg.remove_paths(img,
                         np.int32(img.shape[1]),
                         np.int32(img.shape[0]),
                         stable_paths,
                         np.int32(stable_paths.shape[1]),
                         np.int32(2), # XXX
                         pixel_sum,
                         global_size=(len(stable_paths),), local_size=(1,))
        return pixel_sum

    def all_staff_paths(self):
        img = self.page.img.copy()
        all_paths = []
        threshold = None
        while True:
            self.weights = None
            paths = self.get_stable_paths(img)
            sums = self.validate_and_remove_paths(img, paths).get()
            if threshold is None:
                try:
                    threshold = int(np.median(sums[sums > 0]) * 0.8)
                    all_paths.append(paths.get()[sums >= threshold])
                except ValueError: # all paths have 0 dark pixels
                    return np.empty((0, 2048), np.int32)
            else:
                valid = sums >= threshold
                if not valid.any():
                    return np.concatenate(all_paths) * 2 # XXX: path_scale
                else:
                    all_paths.append(paths.get()[valid])

    def get_staves(self):
        staff_paths = self.all_staff_paths()
        # Sort the path y's in each column, which prevents paths crossing
        staff_paths = np.sort(staff_paths, axis=0)

        staff_lines = []
        cur_staff = []
        last_line_pos = None
        for line in staff_paths:
            # TODO: determine actual extent of staff
            x0 = 0
            x1 = len(line)-1
            xs = np.arange(x0*2, x1*2+1, 2) # XXX: path scale
            staff_line = np.empty((len(xs), 2), int)
            staff_line[:, 0] = xs
            staff_line[:, 1] = line[x0:x1+1] # XXX: path scale
            line_pos = np.median(staff_line[:, 1])
            
            if (not cur_staff
                or line_pos - last_line_pos < self.page.staff_dist*2):
                cur_staff.append(staff_line)
            elif cur_staff:
                # end of staff
                if len(cur_staff) != 5:
                    logging.info('Throwing out staff with %d lines' % len(cur_staff))
                else:
                    staff_lines.append(cur_staff)
                cur_staff = [staff_line]
            last_line_pos = line_pos
        if cur_staff:
            # end of staff
            if len(cur_staff) != 5:
                logging.info('Throwing out staff with %d lines' % len(cur_staff))
            else:
                staff_lines.append(cur_staff)
        if not staff_lines:
            self.staves = np.empty((0, 2, 2), int)
            return self.staves
        staff_center_lines = []
        for lines in staff_lines:
            line = np.array(lines).mean(axis=0)
            keep_points = np.zeros(len(line), bool)
            keep_points[[0, -1]] = 1
            keep_points[1:-1] = (np.diff(line[:-1, 1]) != 0) | (np.diff(line[1:, 1]) != 0)
            line_med = np.median(line[:,1])
            for i in xrange(len(line)):
                keep_points[i] &= np.abs(line[i,1] - line_med) < self.page.staff_dist*2
            staff_center_lines.append(line[keep_points])
        width = max([len(line) for line in staff_center_lines])
        mask = map(lambda line:
                   np.vstack([np.zeros((len(line), 2), bool),
                              np.ones((width-len(line), 2), bool)]),
                   staff_center_lines)
        pad_centers = map(lambda line:
                          np.vstack([line,
                                  -np.ones((width-len(line), 2), int)]),
                          staff_center_lines)
        self.staves = np.ma.array(pad_centers, np.int32, mask=mask, fill_value=-1)
        self.weights = None
        return self.staves
