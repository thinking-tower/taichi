import taichi as ti
import numpy as np
import pytest

if ti.has_pytorch():
    import torch

# minimum int bits without sign
# https://en.cppreference.com/w/cpp/language/types
bits = 16 - 1


@ti.all_archs
@pytest.mark.parametrize('size', [[
    (1 << bits)
], [(1 << i)
    for i in range(bits // 3 - 5, bits // 3)], [(1 << bits // 8) - 1] * 8])
def test_get_external_tensor_shape_access_numpy(size):
    @ti.kernel
    def func(x: ti.ext_arr(), index: ti.template()) -> ti.i32:
        return x.shape[index]

    x_hat = ((1 << bits // 8) - 1) * np.ones(size=size, dtype=np.int32)
    for idx, y_ref in enumerate(size):
        y_hat = func(x_hat, idx)
        assert y_hat == y_ref, "Size of axis {} should equal {} and not {}.".format(
            idx, y_ref, y_hat)


@ti.all_archs
@pytest.mark.parametrize('size', [[1, 1], [1 << bits // 5, 1 << bits // 5],
                                  [1 << bits // 3, 1 << bits // 3]])
def test_get_external_tensor_shape_sum_numpy(size):
    @ti.kernel
    def func(x: ti.ext_arr()) -> ti.i32:
        y = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y += x[i, j]
        return y

    x_hat = (1 << bits // 3) * np.ones(size=size, dtype=np.int32)
    x_ref = x_hat
    y_hat = func(x_hat)
    y_ref = x_ref.sum()
    assert y_ref == y_hat, "Output should equal {} and not {}.".format(
        y_ref, y_hat)


@ti.torch_test
@ti.all_archs
@pytest.mark.parametrize('size', [[
    (1 << bits)
], [(1 << i)
    for i in range(bits // 3 - 5, bits // 3)], [(1 << bits // 8) - 1] * 8])
def test_get_external_tensor_shape_access_numpy(size):
    @ti.kernel
    def func(x: ti.ext_arr(), index: ti.template()) -> ti.i32:
        return x.shape[index]

    x_hat = (1 << bits // 8) * torch.ones(size=size, dtype=torch.int32)
    for idx, y_ref in enumerate(size):
        y_hat = func(x_hat, idx)
        assert y_hat == y_ref, "Size of axis {} should equal {} and not {}.".format(
            idx, y_ref, y_hat)


@ti.torch_test
@ti.all_archs
@pytest.mark.parametrize('size', [[1, 1], [1 << bits // 5, 1 << bits // 5],
                                  [1 << bits // 3, 1 << bits // 3]])
def test_get_external_tensor_shape_sum_numpy(size):
    @ti.kernel
    def func(x: ti.ext_arr()) -> ti.i32:
        y = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y += x[i, j]
        return y

    x_hat = (1 << bits // 3) * torch.ones(size=size, dtype=torch.int32)
    x_ref = x_hat
    y_hat = func(x_hat)
    y_ref = x_ref.sum()
    assert y_ref == y_hat, "Output should equal {} and not {}.".format(
        y_ref, y_hat)
