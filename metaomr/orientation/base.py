import numpy as np

from ..gpu import *
from .. import bitimage

prg = build_program('orientation')

class BaseRotation:
    page = None
    # 3x3 transformation matrix, currently only rotation
    # Translation is done after rotating about top-left corner so that
    # no part of the image is cropped
    transformation = None
    size = None

    def __init__(self, page):
        self.page = page

    def rotation_transformation(self, theta):
        """ Returns the transformation matrix and bounds of the image
            required to not crop the original image. """
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], np.float32)
        corners = [[[0],[0]],
                   [[0],[self.page.orig_size[0]]],
                   [[self.page.orig_size[1]],[0]],
                   [[self.page.orig_size[1]],[self.page.orig_size[0]]]]
        bounds = [np.nan, np.nan, np.nan, np.nan]
        for corner in corners:
            new_point = np.dot(R, corner)
            # any comparison with nan is false
            if not (new_point[0,0] > bounds[0]):
                bounds[0] = new_point[0,0]
            if not (new_point[1,0] > bounds[1]):
                bounds[1] = new_point[1,0]
            if not (new_point[0,0] < bounds[2]):
                bounds[2] = new_point[0,0]
            if not (new_point[1,0] < bounds[3]):
                bounds[3] = new_point[1,0]
        return (np.bmat([[R, [[-bounds[0]], [-bounds[1]]]],
                         [np.mat('0. 0. 1.')]]),
                (int(bounds[3]-bounds[1]), int(bounds[2]-bounds[0])))

    def transform_page(self, M, size):
        pad_size, new_size = self.page.padded_size(size)
        new_img = thr.empty_like(Type(np.uint8, (pad_size[0], pad_size[1]/8)))
        new_img.fill(0)
        # inverse transformation to map output points to input
        Minv = thr.to_device(np.linalg.inv(M).astype(np.float32))
        prg.transform(new_img, self.page.img,
                      np.int32(self.page.img.shape[1]),
                      np.int32(self.page.img.shape[0]),
                      Minv,
                      global_size=new_img.shape[::-1])
        self.page.initial_size = self.page.orig_size
        self.page.orig_size = new_size
        self.page.size = pad_size
        self.page.img = new_img
        self.page.byteimg = bitimage.as_hostimage(new_img)

    def get_transformation(self):
        " Default implementation just uses rotation" 
        self.transformation, self.size = self.rotation_transformation(
                                            self.get_rotation())

    def __call__(self):
        if self.transformation is None:
            self.get_transformation()
            self.transform_page(self.transformation, self.size)
