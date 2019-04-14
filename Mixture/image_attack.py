import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from models import resnet
# from Mixture import utils
import torchvisions.models as models


def fgsm_attack(image, epsilon, data_grad):
	# Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def tensor2variable(x=None, device=None, requires_grad=False):
    """

    :param x:
    :param device:
    :param requires_grad:
    :return:
    """
    x = x.to(device)
    return Variable(x, requires_grad=requires_grad)

def perturbation(samples, ys, model, eplison, device):
        """

        :param samples:
        :param ys:
        :param device:
        :return:
        """
        # ML: add renomalization
        copy_samples = np.copy(samples)[0]
        copy_samples = copy_samples.transpose((1, 0, 2, 3))

        mean = np.array([114.7748, 107.7354, 99.4750]).reshape((1, 3, 1, 1))
        # range [0, 1]
        copy_samples += mean
        copy_samples /= 255.
        # nomalization in ImageNet form
        mean_image = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        std_image = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        copy_samples -= mean_image
        copy_samples /= std_image
        # define loss function
        loss_fun = torch.nn.CrossEntropyLoss()

        var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
        # var_ys = tensor2variable(torch.LongTensor(ys), device=device)

        model.eval()
        preds = model(var_samples)
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(preds, torch.max(preds, 1)[1])
        loss.backward()
        gradient_sign = var_samples.grad.data.cpu().sign().numpy()

        adv_samples = copy_samples + epsilon * gradient_sign
        # renomalization of Imagenet format to range [0, 1]
        adv_samples += mean_image
        adv_samples *= std_image
        # to range [0, 255]
        adv_samples *= 255.
        adv_samples = np.clip(adv_samples, 0.0, 225.0)
        adv_samples -= mean
        samples[0] = adv_samples

        return samples


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--clean_path', default='UCF_inputs.npy', type=str, help='the path for clean examples')
	parser.add_argument(
		'--label_path', default='UCF_labels.npy', type=str, help='the path for clean eamamples labels')
	parser.add_argument(
		'--epsilon', default=0.1, type=float, help='the attack strength of FGSM')

    args = parser.parser_args()
	# load the pretrained model from torchvision
	resnet18 = models.resnet18(pretrained=True).cuda()

	# load clean examples
	clean_stack = np.load(args.clean_path)
	labels = np.load(args.label_path)

	adv_sample = []
    for vid in range(len(clean_stack)):
        print('\r===> in idx {:>4}({:>4} in total) videos are perturbed ... '.format(vid, len(clean_stack)), end=' ')

        batch_adv_images = perturbation(clean_stack[vid:vid+1], labels[vid:vid+1], resnet18, args.epsilon, 'cuda')
        adv_sample.extend(batch_adv_images)

    adv_sample = np.array(adv_sample)






