#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/7 23:05
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : AttackUtils.py
# **************************************



import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


def tensor2variable(x=None, device=None, requires_grad=False):
    """

    :param x:
    :param device:
    :param requires_grad:
    :return:
    """
    x = x.to(device)
    return Variable(x, requires_grad=requires_grad)


def predict(model=None, samples=None, device=None):
    """

    :param model:
    :param samples:
    :param device:
    :return:
    """
    model.eval()
    model = model.to(device)
    copy_samples = np.copy(samples)
    var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
    predictions = model(var_samples.float())
    return predictions

def predict_batch(model=None, samples=None, batch=None, device=None):
    predictions = []
    number_batch = int(math.ceil(len(samples) / batch))
    for index in range(number_batch):
        start = index * batch
        end = min((index + 1) * batch, len(samples))
        print('\r===> predicting adversarial examples in batch {:>2}/{:>4}...'.format(index+1, number_batch))
        predcitions_batch = predict(model, samples[start:end], device)
        predictions.extend(predcitions_batch)

    return predictions