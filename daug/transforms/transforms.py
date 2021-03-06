import numpy as np


def build_stretch_matrix(x, y):
    stretch_transform = np.array(
        [[x, 0, 0],
         [0, y, 0],
         [0, 0, 1]],
        dtype=np.float32)
    return stretch_transform


def build_shear_matrix(x, y):
    tan_x = np.tan(x)
    tan_y = np.tan(y)
    shear_transform = np.array(
        [[1,     tan_x, 0],
         [tan_y, 1,     0],
         [0,     0,     1]],
        dtype=np.float32)
    return shear_transform


def build_rotate_matrix(t):
    cos_t = np.cos(t)  # t must be in radians
    sin_t = np.sin(t)
    rotate_transform = np.array(
        [[cos_t,  sin_t, 0],
         [-sin_t, cos_t, 0],
         [0,          0, 1]],
        dtype=np.float32)
    return rotate_transform


def build_flip_matrix(h, v):
    # reflect about x and y
    if h and v:
        flip_transform = np.array(
            [[-1,  0, 0],
             [ 0, -1, 0],
             [ 0,  0, 1]], dtype=np.float32)
    # reflect about only x
    elif h and not v:
        flip_transform = np.array(
            [[ 1,  0, 0],
             [ 0, -1, 0],
             [ 0,  0, 1]], dtype=np.float32)
    # reflect about only y
    elif not h and v:
        flip_transform = np.array(
            [[-1,  0, 0],
             [ 0,  1, 0],
             [ 0,  0, 1]], dtype=np.float32)
    # do not reflect
    else:
        flip_transform = np.eye(3, dtype=np.float32)

    return flip_transform


def build_translate_matrix(x, y):
    translate_transform = np.array(
        [[1, 0, x],
         [0, 1, y],
         [0, 0, 1]], dtype=np.float32)
    return translate_transform


def build_transformation_matrix(
        imsize, theta=None, offset=None, flip=None, shear=None, stretch=None):

    # use the identity matrix as default
    if theta is None:
        theta = 0.
    if offset is None:
        offset = (0., 0.)
    if flip is None:
        flip = (False, False)
    if shear is None:
        shear = (0., 0.)
    if stretch is None:
        stretch = (1., 1.)

    cx, cy = np.array(imsize) / 2 - 0.5
    center_matrix = build_translate_matrix(cx, cy)
    stretch_matrix = build_stretch_matrix(*stretch)
    shear_matrix = build_shear_matrix(*shear)
    rotate_matrix = build_rotate_matrix(theta)
    flip_matrix = build_flip_matrix(*flip)
    uncenter_matrix = build_translate_matrix(-cx, -cy)
    translate_matrix = build_translate_matrix(*offset)

    transform_matrix = center_matrix.dot(
        stretch_matrix).dot(
        shear_matrix).dot(
        rotate_matrix).dot(
        flip_matrix).dot(
        uncenter_matrix).dot(
        translate_matrix
    )

    return transform_matrix
