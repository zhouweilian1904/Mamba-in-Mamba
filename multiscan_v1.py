import numpy as np
import torch
from einops import rearrange


def direction1(input_HW):
    h = input_HW.shape[0]
    w = input_HW.shape[1]
    result = []
    for i in range(h):
        if (i + 1) % 2 == 1:
            result.extend(input_HW[i])
        else:
            result.extend(input_HW[i][::-1])
    return result


def direction2(input_HW):
    h = input_HW.shape[0]
    w = input_HW.shape[1]
    result = []
    for i in range(w):
        if (i + 1) % 2 == 1:
            result.extend(input_HW[:, i])
        else:
            result.extend(input_HW[:, i][::-1])
    return result


def direction3(input_HW):
    h = input_HW.shape[0]
    w = input_HW.shape[1]
    result = []
    for i in range(h):
        if (i + 1) % 2 == 0:
            result.extend(input_HW[h - 1 - i])
        else:
            result.extend(input_HW[h - 1 - i][::-1])
    return result


def direction4(input_HW):
    h = input_HW.shape[0]
    w = input_HW.shape[1]
    result = []
    for i in range(w):
        if (i + 1) % 2 == 0:
            result.extend(input_HW[:, w - 1 - i])
        else:
            result.extend(input_HW[:, w - 1 - i][::-1])
    return result


def direction5(input_HW):
    h = input_HW.shape[0]
    w = input_HW.shape[1]
    result = []
    for i in range(h):
        if (i + 1) % 2 == 0:
            result.extend(input_HW[i])
        else:
            result.extend(input_HW[i][::-1])
    return result


def direction6(input_HW):
    h = input_HW.shape[0]
    w = input_HW.shape[1]
    result = []
    for i in range(w):
        if (i + 1) % 2 == 1:
            result.extend(input_HW[:, w - 1 - i])
        else:
            result.extend(input_HW[:, w - 1 - i][::-1])
    return result


def direction7(input_HW):
    h = input_HW.shape[0]
    w = input_HW.shape[1]
    result = []
    for i in range(h):
        if (i + 1) % 2 == 1:
            result.extend(input_HW[h - 1 - i])
        else:
            result.extend(input_HW[h - 1 - i][::-1])
    return result


def direction8(input_HW):
    h = input_HW.shape[0]
    w = input_HW.shape[1]
    result = []
    for i in range(w):
        if (i + 1) % 2 == 0:
            result.extend(input_HW[:, i])
        else:
            result.extend(input_HW[:, i][::-1])
    return result


method_map = {1: direction1, 2: direction2, 3: direction3, 4: direction4, 5: direction5, 6: direction6, 7: direction7,
              8: direction8}


# img_test: batch size,H,W,C
# method_index:方法下标：1-8
def multiscan_v1(img_test, method_index):
    input = img_test.numpy()
    batchsize = input.shape[0]
    h = img_test.shape[1]
    w = input.shape[2]
    c = input.shape[3]
    # print(h, w, c)
    sample_all = []
    for i in range(batchsize):
        sample_pic = []
        for j in range(c):
            sample_pic.append(method_map[method_index](input[i][:, :, j]))
        sample_all.append(sample_pic)
    sample_all_array = np.array(sample_all)
    sample_all_torch = torch.from_numpy(sample_all_array)
    sample_all_rearrange = rearrange(sample_all_torch, 'b c s -> b s c')
    return sample_all_rearrange


# batchsize:2 H:2 w:3 c:2
# img_test = torch.randn(1, 5, 5, 2)
# print(img_test.shape, img_test)
# result = multiscan(img_test, 4)
# print(result.shape, result)
# print("trans shape:", result.shape)
