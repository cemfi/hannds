import numpy as np

from train_nn import *


def test_filter_noaction():
    zero = np.zeros((10, 10))
    new_zero = majority_filter(zero)
    assert (zero == new_zero).all()
    new_zero = causal_filter(zero)
    assert (zero == new_zero).all()

    zero[0, 1] = 1
    zero[1, 1] = 1
    zero[0, 5] = 2
    zero[2, 2] = 2
    zero[3, 3] = 1
    zero[4, 4] = 2
    zero[5, 5] = 1
    zero[6, 5] = 1
    zero[6, 5] = 1
    zero[6, 6] = 2
    zero[7, 6] = 2
    zero[8, 6] = 2

    new_zero = majority_filter(zero)
    assert (zero == new_zero).all()
    new_zero = causal_filter(zero)
    assert (zero == new_zero).all()


def test_causal_action():
    zero = np.zeros((5, 3))
    zero[1, 0] = 1
    zero[2, 0] = 2
    zero[3, 0] = 2
    zero[0, 1] = 2
    zero[1, 1] = 1
    zero[3, 2] = 2
    zero[4, 2] = 1
    zero = causal_filter(zero)

    correct = torch.zeros((5, 3))
    correct[1, 0] = 1
    correct[2, 0] = 1
    correct[3, 0] = 1
    correct[0, 1] = 2
    correct[1, 1] = 2
    correct[3, 2] = 2
    correct[4, 2] = 2
    assert (zero == correct).all()
