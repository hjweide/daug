import cv2
import numpy as np
from daug.transforms import build_transformation_matrix


# TODO: test shear, flip, and translate

def run_rotation_scale_tests(n=10):
    heights = np.random.randint(16, 512, size=n)
    widths = np.random.randint(16, 512, size=n)
    thetas = (2 * np.pi)  * np.random.random(n)
    scales = np.random.random(n)
    for i in range(n):
        theta, scale = thetas[i], scales[i]
        height, width = heights[i], widths[i]

        # getRotMat2D only supports isotropic scaling
        L = cv2.getRotationMatrix2D(
            (height / 2, width / 2), 180. / np.pi * theta, scale)
        M = build_transformation_matrix(
            (height, width), theta=theta, stretch=(scale, scale))

        assert np.allclose(L, M[:2, :]), (
            'OpenCV transformation matrix:\n'
            '%r,\nbut Daug transformation matrix:\n%r' % (L, M[:2, :]))
    print('All rotation/scale tests passed.')


def main():
    run_rotation_scale_tests(n=10)


if __name__ == '__main__':
    main()
