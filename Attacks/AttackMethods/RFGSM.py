#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References: F. Tram`er, et al.,"Ensemble adversarial training: Attacks and defenses," in ICLR, 2018.
# Reference Implementation from Authors (TensorFlow): https://github.com/ftramer/ensemble-adv-training
# **************************************
# @Time    : 2018/9/8 0:30
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : RFGSM.py
# **************************************

import math

import numpy as np
import torch

from Attacks.AttackMethods.AttackUtils import tensor2variable
from Attacks.AttackMethods.attacks import Attack


class RFGSMAttack(Attack):

    def __init__(self, model=None, epsilon=None, alpha=None):
        """

        :param model:
        :param eps:
        :param alpha:
        """
        super(RFGSMAttack, self).__init__(model)
        self.model = model

        self.epsilon = epsilon
        self.alpha = alpha

    def perturbation(self, samples, ys, device):
        """

        :param samples:
        :param ys:
        :param device:
        :return:
        """
        mean = np.array([114.7748, 107.7354, 99.4750]).reshape((1, 3, 1, 1, 1))
        copy_samples = np.copy(samples)

        # add randomized single-step attack
        copy_samples = copy_samples + (self.alpha * self.epsilon * np.sign(np.random.randn(*copy_samples.shape)))
        copy_samples = copy_samples.astype(np.float32)

        eps = (1.0 - self.alpha) * self.epsilon
        var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
        var_ys = tensor2variable(torch.LongTensor(ys), device=device)

        self.model.eval()
        preds = self.model(var_samples)
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(preds, torch.max(var_ys, 1)[1])
        loss.backward()

        gradient_sign = var_samples.grad.data.cpu().sign().numpy()
        adv_samples = copy_samples + eps * gradient_sign

        adv_samples += mean
        adv_samples = np.clip(adv_samples, 0.0, 255.0)
        adv_samples -= mean
        # adv_samples = np.clip(adv_samples, 0.0, 1.0)
        return adv_samples

    def batch_perturbation(self, xs, ys, batch_size, device):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param device:
        :return:
        """
        assert len(xs) == len(ys), "The lengths of samples and its ys should be equal"

        adv_sample = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')

            batch_adv_images = self.perturbation(xs[start:end], ys[start:end], device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
