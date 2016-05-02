import cv2
import numpy as np

from daug.transforms import build_transformation_matrix


def transform_image(X, M):
    assert M.shape == (3, 3), 'expected (3, 3) matrix, got %r' % (M.shape,)
    assert X.dtype == np.float32, 'expected dtype float32, got %r' % (X.dtype)
    dsize = X.shape[::-1]
    return cv2.warpAffine(X, M[:2, :], dsize, flags=cv2.WARP_INVERSE_MAP)


def transform_minibatch(X, **kwargs):
    Xtr = np.empty(X.shape, dtype=X.dtype)  # copy to avoid accumulating errors
    # TODO: there is certainly a better way to handle different
    # number of channels
    for i in range(X.shape[0]):
        if X[i].shape[0] == 1:
            M = build_transformation_matrix(X[i, 0].shape, **kwargs)
            Xtr[i, 0] = transform_image(X[i, 0], M)
        elif X[i].shape[0] == 3:
            M = build_transformation_matrix(X[i].shape[1:3], **kwargs)
            Xtr[i, 0] = transform_image(X[i, 0], M)
            Xtr[i, 1] = transform_image(X[i, 1], M)
            Xtr[i, 2] = transform_image(X[i, 2], M)
        else:
            raise NotImplementedError(
                'Images with %d channels are not yet supported.' % (
                    X[1].shape[1]))

    return Xtr
