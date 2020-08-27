import taichi as ti
import numpy as np


@ti.all_archs
def test_get_external_tensor_shape_access():
    @ti.kernel
    def func(x: ti.ext_arr(), index: ti.template()) -> ti.i32:
        return x.shape[index]

    size = np.random.randint(low=1, high=10, size=(8, ))
    x = np.random.randint(low=0, high=np.prod(size), size=size, dtype=np.int32)
    for idx, i in enumerate(size):
        assert func(x, idx) == i, "Axis {} should equal {}.".format(idx, i)


@ti.all_archs
def test_get_external_tensor_shape_sum():
    @ti.kernel
    def func(x: ti.ext_arr()) -> ti.i32:
        y = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y += x[i, j]
        return y

    size = np.random.randint(low=1, high=10, size=(2, ))
    x = np.random.randint(low=0, high=np.prod(size), size=size, dtype=np.int32)
    out = func(x)
    assert out == x.sum(), "{} should equal output {}.".format(out, y)
