from ..gpu import *
import moonshine.bitimage, moonshine.util
import numpy as np
import logging
try:
    import scipy.signal as scipy_signal
except ImportError:
    scipy_signal = None

prg = build_program("staves")

class BaseStaves(object):
    page = None
    staves = None
    nostaff_img = None

    def __init__(self, page):
        self.page = page

    def staff_center_filter(self):
        output = thr.empty_like(self.page.img)
        prg.staff_center_filter(self.page.img,
                                np.int32(self.page.staff_thick),
                                np.int32(self.page.staff_dist),
                                output,
                                global_size=self.page.img.shape[::-1])
        return output

    def __call__(self):
        if self.staves is None:
            self.get_staves()
            if not isinstance(self.staves, np.ma.masked_array):
                self.staves = np.ma.array(self.staves, fill_value=-1)
        return self.staves

    def nostaff(self):
        if self.nostaff_img is None:
            self.remove_staves()
        return self.nostaff_img

    def get_staves(self):
        NotImplementedError("Use a concrete Staves subclass.")

    def remove_staff_gaps(self, staff, max_gap=None):
        if max_gap is None:
            max_gap = 8 * self.page.staff_dist

        if hasattr(staff, 'compressed'):
            staff = staff.compressed().reshape((-1, 2))
        staff = staff[staff[:,0] >= 0]
        is_gapped = np.empty(staff.shape[0], bool)
        is_gapped[0] = True
        is_gapped[1:] = np.diff(staff[:,0]) > max_gap
        gap_inds, = np.where(is_gapped)
        gap_starts = gap_inds
        gap_ends = list(gap_inds[1:]) + [None]

        segments = []
        lengths = []
        for start, end in zip(gap_starts, gap_ends):
            segment = staff[start:end]
            segments.append(segment)
            lengths.append(segment[-1, 0] - segment[0, 0])
        return segments[np.argmax(lengths)]

    def refine_and_remove_staves(self, refine_staves=False, remove_staves=True,
                                 staves=None, img=None):
        assert refine_staves or remove_staves, 'Need something to do'
        if staves is None:
            staves = self()
        if not len(staves):
            # This function does nothing if there are no staves
            return staves, self.page.img.copy()
        if img is None:
            img = self.page.img
        if refine_staves:
            refined_num_points = np.int32(self.page.orig_size[1] // 8)
            refined_staves = thr.empty_like(Type(np.int32,
                                (staves.shape[0], refined_num_points, 2)))
            refined_staves.fill(-1)
        else:
            refined_num_points = np.int32(0) # disable refined_staves
            refined_staves = thr.empty_like(Type(np.int32, 1)) # dummy array
        if remove_staves:
            nostaff_img = img.copy()
        else:
            nostaff_img = img
            refined_num_points = np.int32(-refined_num_points)
        prg.staff_removal(thr.to_device(staves.filled().astype(np.int32)),
                          np.int32(self.page.staff_thick+1),
                          np.int32(self.page.staff_dist),
                          nostaff_img,
                          np.int32(nostaff_img.shape[1]),
                          np.int32(nostaff_img.shape[0]),
                          refined_staves,
                          refined_num_points,
                          global_size=staves.shape[1::-1],
                          local_size=(staves.shape[1], 1))
        if refine_staves:
            if not (refined_staves != -1).any():
                return np.ma.array(np.empty([0, 2], np.int32)), nostaff_img
            new_staves = map(self.remove_staff_gaps, refined_staves.get())
            # Must move all (-1, -1) points to end of each staff
            num_points = max([staff.shape[0] for staff in new_staves])
            staves_copy = np.empty((staves.shape[0], num_points, 2), np.int32)
            mask = np.ones_like(staves_copy, dtype=bool)
            for i, staff in enumerate(new_staves):
                # Clean up single spurious points (requires scipy)
                if scipy_signal is None:
                    logging.warn('Scipy not installed; staff refinement will be'
                                 ' poor quality')
                else:
                    staff[:, 1] = scipy_signal.medfilt(staff[:, 1],
                                    -(-(self.page.staff_dist * 4 / 8) & -2) + 1)
                staves_copy[i, :len(staff)] = staff
                mask[i, :len(staff)] = 0
            order = np.argsort(staves_copy[:, 0, 1]) # sort by y0
            staves_copy = staves_copy[order]
            mask = mask[order]
            staves = np.ma.array(staves_copy, mask=mask, fill_value=-1)
        return staves, nostaff_img

    def remove_staves(self, refine_staves=False):
        """ Default staff removal implementation, with optional refinement """
        self() # must have staves
        self.staves, self.nostaff_img = self.refine_and_remove_staves(
                remove_staves=True, refine_staves=refine_staves)

    def extract_staff(self, staff, img=None, extract_lines=4):
        if type(staff) is int:
            staff = self()[staff]
        if hasattr(staff, 'mask'):
            staff = staff.compressed().reshape([-1, 2])
        if img is None:
            img = self.page.img
        output = thr.empty_like(Type(np.uint8,
                    (self.page.staff_dist*extract_lines + 1,
                     self.page.orig_size[1]/8)))
        output.fill(0)
        prg.extract_staff(thr.to_device(staff.astype(np.int32)),
                          np.int32(staff.shape[0]),
                          np.int32(self.page.staff_dist),
                          img,
                          np.int32(img.shape[1]),
                          np.int32(img.shape[0]),
                          output,
                          global_size=output.shape[::-1])
        return output

    def extend_staff(self, staff):
        """ Extract a piece of the staff along the center line, then extend
            the staff on either side until we reach a gap """
        staff_img = moonshine.bitimage.as_hostimage(self.extract_staff(staff,
                                                        extract_lines=1))
        is_dark = staff_img.any(axis=0)
        components, n_components = moonshine.util.label_1d(is_dark)
        staff_points = self()[staff].compressed().reshape((-1, 2))
        c0 = components[staff_points[0, 0]]
        if c0:
            staff_min = np.where(components == c0)[0][0]
            if staff_min != staff_points[0, 0]:
                staff_points = np.r_[[[staff_min, staff_points[0, 1]]],
                                     staff_points]
        c1 = components[staff_points[-1, 0]]
        if c1:
            staff_max = np.where(components == c1)[0][-1]
            if staff_max != staff_points[-1, 0]:
                staff_points = np.r_[staff_points,
                                     [[staff_max, staff_points[-1, 1]]]]
        return staff_points

    def extend_staves(self):
        if not len(self.staves):
            return # This does nothing if there are no staves
        new_staves = [self.extend_staff(i) for i in xrange(len(self.staves))]
        num_segments = max([s.shape[0] for s in new_staves])
        staves = np.ma.empty((len(new_staves), num_segments, 2),
                             dtype=np.int32,
                             fill_value=-1)
        staves.mask = np.ones_like(staves, dtype=bool)
        for i, staff in enumerate(new_staves):
            staves[i, :len(staff)] = staff
            staves.mask[i, :len(staff)] = False
        self.staves = staves

    def score(self, labeled_staves):
        staff_med = np.ma.median(self()[..., 1], axis=1)
        label_med = np.ma.median(labeled_staves[..., 1], axis=1)
        matches = moonshine.util.match(staff_med, label_med)
        was_found = np.zeros(len(label_med), bool)
        is_correct = (np.abs(staff_med - label_med[matches])
                        < self.page.staff_dist)
        for i in xrange(len(is_correct)):
            was_found[matches[i]] |= is_correct[i]
        sensitivity = float(was_found.sum()) / len(label_med)
        specificity = 0
        # Can have at most 1 true match for each labeled staff
        for ind in xrange(len(label_med)):
            specificity += is_correct[matches == ind].any()
        specificity = float(specificity) / max(1, len(staff_med))
        return sensitivity, specificity

    def scaled_staff(self, staff_num):
        """ Scale each staff so that staff_dist ~= 6, and scale horizontally
            by a factor of 1 / max(1, staff_thick/2). """
        scale_x = 1.0 / max(1, (self.page.staff_thick + 1) // 2)
        scale_y = 6.0 / float(self.page.staff_dist)
        extracted = self.extract_staff(staff_num, self.page.img)
        scaled_img = moonshine.bitimage.scale(extracted, scale_x, scale_y)
        return scaled_img[:24], scale_x, scale_y

    def get_staff(self, staff_num):
        if hasattr(self(), 'compressed'):
            return self()[staff_num].compressed().reshape((-1, 2))
        else:
            return self()[staff_num]
    def staff_y(self, staff_num, x):
        staff = self.get_staff(staff_num)
        if (staff[:,0] < x).all():
            # Extrapolate from overall direction of staff
            return staff[0,1] + ((staff[0,0] - x)
                                 * (staff[0,1] - staff[-1,1])
                                 / (staff[-1,0] - staff[0,0]))
        elif (staff[:,0] > x).all():
            return staff[-1,1] + ((x - staff[-1,0])
                                  * (staff[-1,1] - staff[0,1])
                                  / (staff[-1,0] - staff[0,0]))
        # First staff point past x
        past_x = np.argmin(staff[:, 0] < x)
        if staff[past_x, 0] == x:
            return staff[past_x, 1]
        else:
            assert past_x > 0
            pre_x = past_x - 1
            return staff[past_x, 1] + ((x - staff[pre_x, 0])
                                        * (staff[past_x,1] - staff[pre_x,1])
                                        / (staff[past_x,0] - staff[pre_x,0]))
    def show(self):
        import pylab as p
        for staff in self():
            xs = staff[:, 0].compressed()
            ys = staff[:, 1].compressed()
            p.plot(xs, ys, 'g')
