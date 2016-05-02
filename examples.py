import sys
from daug.utils import transform_minibatch
import cv2
import numpy as np

from os import listdir
from os.path import join, isdir, isfile


def load_cifar10():
    import cPickle as pickle
    import tarfile
    path = join('cifar10', 'cifar-10-python.tar.gz')
    if not isfile(path):
        raise IOError('Please download the cifar10 data set to %s.' % path)

    with tarfile.open(path, 'r:gz') as f:
        f.extractall(path='cifar10')
    path = join('cifar10', 'cifar-10-batches-py', 'data_batch_1')
    with open(path, 'rb') as f:
        X = pickle.load(f)['data']

    X = X.reshape(-1, 3, 32, 32).astype(np.float32)
    X /= 255.
    X = X[:, ::-1, :, :]  # RGB -> BGR to be compatible with OpenCV

    return X


def load_mnist():
    import gzip
    path = join('mnist', 'train-images-idx3-ubyte.gz')
    if not isfile(path):
        raise IOError('Please download the mnist data set to %s.' % path)
    with gzip.open(path) as f:
        X = np.frombuffer(f.read(), np.uint8, offset=16)
        X = X.reshape(-1, 1, 28, 28).astype(np.float32)
        X /= 255.

    return X


def load_other(datadir, imsize):
    if datadir is None or not isdir(datadir):
        raise IOError('Please place the images in %s.' % datadir)
    if imsize is None or imsize < 0:
        raise ValueError('%r is not a valid image size.' % imsize)

    filenames = listdir(datadir)
    filepaths = [join(datadir, fname) for fname in filenames]
    X = np.empty((len(filepaths), 3, imsize, imsize), dtype=np.float32)

    for i, fpath in enumerate(filepaths):
        img = cv2.imread(fpath).astype(np.float32) / 255.
        X[i] = cv2.resize(img, (imsize, imsize)).transpose(2, 0, 1)

    return X


def main(example, imsize=None):
    if example == 'mnist':
        X = load_mnist()
    elif example == 'cifar10':
        X = load_cifar10()
    else:
        X = load_other(example, imsize)

    bs = 8  # number of images to show at a time
    num_batches = (X.shape[0] + bs - 1) / bs
    for i in range(num_batches):
        # get the minibatch
        idx = slice(i * bs, (i + 1) * bs)
        Xb = X[idx]

        # this is the relevant part
        Xbt = transform_minibatch(
            Xb,
            offset=(0., 0.), theta=np.pi / 4,
            flip=(True, True),
            shear=(0.0, 0.0), stretch=(1.0, 1.0),
        )

        # this is just for visualization
        original = np.hstack(np.squeeze(Xb.transpose(0, 2, 3, 1)))
        transformed = np.hstack(np.squeeze(Xbt.transpose(0, 2, 3, 1)))
        img = np.vstack((original, transformed))
        img = (img * 255.).astype(np.uint8)
        cv2.imshow('minibatch', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        print('Usage:')
        print(' Provided examples:')
        print('  python examples.py mnist')
        print('  python examples.py cifar10')
        print(' Custom examples (provide a path to a directory of images):')
        print('  python examples.py /home/user/path/to/images')
        print('  python examples.py /home/user/path/to/images 100')
    elif len(sys.argv) == 2:
        example = sys.argv[1]
        main(example)
    else:
        example = sys.argv[1]
        imsize = int(sys.argv[2])  # the images will be resized to imsize
        main(example, imsize)
