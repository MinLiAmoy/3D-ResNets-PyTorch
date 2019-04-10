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

def perturbation(self, samples, ys, device):
        """

        :param samples:
        :param ys:
        :param device:
        :return:
        """
        # ML: add renomalization
        mean = np.array([114.7748, 107.7354, 99.4750]).reshape((1, 3, 1, 1, 1))
        copy_samples = np.copy(samples)
        # range [0, 1]
        copy_samples += mean
        copy_samples /= 255.
        # nomalization in ImageNet form
        mean_image = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1, 1))
        std_image = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1, 1))
        copy_samples -= mean_image
        copy_samples /= std_image

        for vid in range(len(copy_samples)):
        	data = copy_samples[vid]
        	

        var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
        var_ys = tensor2variable(torch.LongTensor(ys), device=device)

        self.model.eval()
        preds = self.model(var_samples)
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(preds, torch.max(var_ys, 1)[1])
        loss.backward()
        gradient_sign = var_samples.grad.data.cpu().sign().numpy()

        adv_samples = copy_samples + self.epsilon * gradient_sign
        adv_samples += mean
        adv_samples = np.clip(adv_samples, 0.0, 225.0)
        adv_samples -= mean
        # adv_samples = np.clip(adv_samples, 0.0, 1.0)

        return adv_samples


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--clean_path', default='UCF_inputs.npy', type=str, help='the path for clean examples')
	parser.add_argument(
		'--label_path', default='UCF_labels.npy', type=str, help='the path for clean eamamples labels')
	parser.add_argument(
		'--sample_rate', default=4, type=int, help='the sample_rate for adv examples, eg. 4-sample once for every 4 frames')

	# load the pretrained model from torchvision
	resnet18 = models.resnet18(pretrained=True).cuda()

	# load clean examples
	clean_stack = np.load(clean_path)
	labels = np.load(label_path)

	adv_sample = []
    number_batch = int(math.ceil(len(clean_stack) / batch_size))
    for index in range(number_batch):
        start = index * batch_size
        end = min((index + 1) * batch_size, len(clean_stack))
        print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')

        batch_adv_images = perturbation(clean_stack[start:end], labels[start:end], 'cuda')
        adv_sample.extend(batch_adv_images)

    adv_sample = np.array(adv_sample)






