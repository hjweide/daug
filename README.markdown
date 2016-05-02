# Data Augmentation

This is an incredibly simple, work-in-progress, **d**ata **aug**mentation library that
can be dropped into any project and used with the 
```(batch, channels, height, width)``` representation of image data.

## Usage
``` bash

    git clone https://github.com/hjweide/daug
    cp -r daug/daug ~/path/to/project

```
Transform an entire minibatch directly:
``` python

>>> from daug.utils import transform_minibatch
>>> X = np.zeros((128, 3, 64, 64), dtype=np.float32)
>>> Xb = transform_minibatch(
           Xb,
           offset=(0., 0.), theta=np.pi / 4.,
           flip=(False, False),
           shear=(0.0, 0.0), stretch=(1.0, 1.0)
    )

```

Or build your own transformation matrix:

``` python 

>>> from daug.transforms import build_transformation_matrix
>>> X = np.zeros((128, 3, 64, 64), dtype=np.float32)
>>> M = build_transformation_matrix(
        X.shape[2:]
        offset=(0., 0.), theta=np.pi / 4.,
        flip=(False, False),
        shear=(0.0, 0.0), stretch=(1.0, 1.0)
    )
>>> print M
>>> [[  0.70710677   0.70710677 -13.25483322]
     [ -0.70710677   0.70710677  32.        ]
     [  0.           0.           1.        ]]
```


## Design Philosophy

1. Simplicity: the user should be able to just copy the ```daug``` directory
into any project.

2. Transparency: the user should be able to access the affine transformation
directly and use it to apply the same transformation to, for example, bounding
box coordinates.

## Dependencies
The only non-standard dependency is ```OpenCV```, which can be downloaded from:
[https://github.com/Itseez/opencv](https://github.com/Itseez/opencv).

Installation instructions are available [here](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation).
