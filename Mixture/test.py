import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from models import resnet
from Mixture import utils


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--adv_path', default='FGSM_AdvExamples_0.1.npy', type=str, help='the path for adversarial examples')
	parser.add_argument(
		'--clean_path', default='UCF_inputs.npy', type=str, help='the path for clean examples')
	parser.add_argument(
		'--sample_rate', default=4, type=int, help='the sample_rate for adv examples, eg. 4-sample once for every 4 frames')

	# load model
	raw_model_location = 'resnet-18-kinetics-ucf101_split1.pth'
	checkpoint = torch.load(raw_model_location)
	raw_model = resnet.resnet18(num_classes=101, shortcut_type='A', sample_size=112, sample_duration=16)
	raw_model = raw_model.cuda()
	raw_model.load_state_dict(checkpoint['state_dict'])



	# subsample the adversarial examples and replace the corresponding ones in clean data
	adv_stack = np.load(adv_path)
	clean_stack = np.load(clean_path)
	mix_stack = np.copy(clean_stack)

	mix_stack[:,:,:sample_rate:,:,:] = adv_stack[:,:,:sample_rate:,:,:]

	# Inference
	mix_labels = predict_batch(model=raw_model, samples=mix_stack, batch=10, device='cuda')
    mix_labels = np.array(mix_labels)
    mix_labels = torch.max(torch.from_numpy(mix_labels), 1)[1]
    mix_labels = mix_labels.cpu().numpy()

    # np.save('{}{}_AdvExamples_{}.npy'.format(self.adv_examples_dir, self.attack_name, self.epsilon/255.), adv_samples)
    # np.save('{}{}_AdvLabels_{}.npy'.format(self.adv_examples_dir, self.attack_name, self.epsilon/255.), adv_labels)
    # np.save('{}{}_TrueLabels_{}.npy'.format(self.adv_examples_dir, self.attack_name, self.epsilon/255.), self.labels_samples)

    mis = 0
    for i in range(len(mix_stack)):
        if mix_labels[i].argmax(axis=0) != adv_labels[i]:
            mis = mis + 1
    print('\nFor **{}** on **{}**: misclassification ratio is {}/{}={:.1f}%\n'.format(self.attack_name, self.dataset, mis, len(adv_samples),
                                                                                          mis / len(adv_labels) * 100))

