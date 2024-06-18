import numpy as np
import torch
from torch import nn
from einops import rearrange


def direction(input_HW, method_index):
    h, w = input_HW.shape[:2]
    is_even_row = np.arange(h) % 2 == 0
    is_even_col = np.arange(w) % 2 == 0
    rows = np.arange(h)
    cols = np.arange(w)

    if method_index == 1 or method_index == 5:
        cols = np.where(is_even_row[:, None], cols, cols[::-1])
    elif method_index == 2 or method_index == 6:
        rows = np.where(is_even_col[None, :], rows, rows[::-1])
    elif method_index == 3 or method_index == 7:
        rows = rows[::-1]
        cols = np.where(is_even_row[:, None], cols, cols[::-1])
    elif method_index == 4 or method_index == 8:
        cols = cols[::-1]
        rows = np.where(is_even_col[None, :], rows, rows[::-1])
    else:
        raise ValueError(f"Invalid method_index: {method_index}")

    if method_index > 4:
        rows, cols = cols, rows

    return input_HW[rows[:, None], cols]


def multiscan_v2(img_test, method_index):
    input = img_test.numpy()
    batchsize, h, w, c = input.shape

    output = np.empty((batchsize, h * w, c), input.dtype)
    for i in range(batchsize):
        for j in range(c):
            output[i, :, j] = direction(input[i, :, :, j], method_index).ravel()

    return torch.from_numpy(output)
