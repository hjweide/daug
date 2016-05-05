import cv2
import numpy as np

from daug.transforms import build_transformation_matrix


def generate_transformations(samples, imsize,
                             rotation=None, offset=None,
                             flip=None, shear=None, stretch=None):
    if rotation is not None:
        theta_min, theta_max = rotation
        rotations = 2 * theta_max * np.random.random(size=samples) - theta_min
    else:
        rotations = [None] * samples

    if offset is not None:
        offset_min, offset_max = offset
        offsets = (offset_max - offset_min) * np.random.random(
            size=2 * samples).reshape(samples, 2) - offset_max
    else:
        offsets = [None] * samples

    if flip is not None:
        p_flip_horizontal, p_flip_vertical = flip
        flip_horizontal = np.random.choice(
            [True, False], size=samples, replace=True,
            p=(p_flip_horizontal, 1 - p_flip_horizontal))

        flip_vertical = np.random.choice(
            [True, False], size=samples, replace=True,
            p=(p_flip_vertical, 1 - p_flip_vertical))
        flips = [(ph, pv) for ph, pv in zip(flip_horizontal, flip_vertical)]
    else:
        flips = [None] * samples

    if shear is not None:
        shear_min, shear_max = shear
        shears = (shear_max - shear_min) * np.random.random(
            size=(2 * samples)).reshape(samples, 2) - shear_max
    else:
        shears = [None] * samples

    if stretch is not None:
        stretch_min, stretch_max = stretch
        log_dist = (np.log(stretch_max) + (np.log(stretch_min) - np.log(
            stretch_min)) * np.random.random(
                size=(2 * samples)).reshape(samples, 2))
        stretches = np.e ** log_dist
    else:
        stretches = [None] * samples

    transformations = []
    for i in range(samples):
        M = build_transformation_matrix(
            imsize, theta=rotations[i], offset=offsets[i],
            flip=flips[i], shear=shears[i], stretch=stretches[i]
        )
        transformations.append(M)

    return np.array(transformations)


def transform_image(X, M):
    assert M.shape == (3, 3), 'expected (3, 3) matrix, got %r' % (M.shape,)
    assert X.dtype == np.float32, 'expected dtype float32, got %r' % (X.dtype)
    dsize = X.shape[:2][::-1]  # get (h, w), flip them for OpenCV

    return cv2.warpAffine(X, M[:2, :], dsize)


def transform_minibatch(X, M):
    assert X.shape[0] == M.shape[0], (
        'need X.shape[0] == M.shape[0], but %d != %d' % (
            X.shape[0], M.shape[0]))

    Xtr = np.empty(X.shape, dtype=X.dtype)  # copy to avoid accumulating errors
    for i in range(X.shape[0]):
        if X[i].shape[0] == 1:
            Xtr[i, 0] = transform_image(X[i, 0], M[i])
        elif X[i].shape[0] == 3:
            # bchw --> bhwc --> bchw
            Xtr[i] = transform_image(
                X[i].transpose(1, 2, 0), M[i]).transpose(2, 0, 1)
        else:
            raise NotImplementedError(
                'Images with %d channels are not yet supported.' % (
                    X[1].shape[1]))

    return Xtr
