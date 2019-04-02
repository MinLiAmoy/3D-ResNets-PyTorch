'''
refer to DEEPSEC:CleanDatasets/CandidatesSelections.py
'''

import torch
from torch.autograd import Variable
import time
import random
import os
import sys
import numpy as np

def candidate_selection(data_loader, model, criterion, opt):
	# print('Start selecting the clean data candidate with number of {}'.format(opt.number_candidate))
	
	successful = []
	model.eval()

	with torch.no_grad():
		for i, (inputs, targets) in enumerate(data_loader):
			print(i)
			if not opt.no_cuda:
				targets = targets.cuda(async=True)
			inputs = Variable(inputs)
			targets = Variable(targets)
			outputs = model(inputs)
			_, predicted = torch.max(outputs.data, 1)
			if predicted == targets:
				_, least_likly_class = torch.min(outputs.data, 1)
				successful.append([inputs, targets, least_likly_class])
	# print(len(successful))
	print('Start selecting the clean data candidate with number of {}/{}'.format(opt.number_candidate, len(successful)))
	# torch.save(successful, 'succ.pt')

	candidates = random.sample(successful, opt.number_candidate)

	candidate_images = []
	candidate_labels = []
	candidate_llc = []
	candidate_targets = []

	for index in range(len(candidates)):
		image = candidates[index][0].cpu().numpy()
		image = np.squeeze(image, axis=0)
		candidate_images.append(image)

		label = candidates[index][1].cpu().numpy()[0]

		if index == 0:
			break
		llc = candidates[index][2].cpu().numpy()[0]

		# selection for the targeted label
		classes = [i for i in range(101)]
		classes.remove(label)
		target = random.sample(classes, 1)[0]

		one_hot_label = [0 for i in range(101)]
		one_hot_label[label] = 1

		one_hot_llc = [0 for i in range(101)]
		one_hot_llc[llc] = 1

		one_hot_target = [0 for i in range(101)]
		one_hot_target[target] = 1

		candidate_labels.append(one_hot_label)
		candidate_llc.append(one_hot_llc)
		candidate_targets.append(one_hot_target)

	candidate_images = np.array(candidate_images)
	candidate_labels = np.array(candidate_labels)
	candidate_llc = np.array(candidate_llc)
	candidate_targets = np.array(candidate_targets)

	np.save('ucf_inputs.npy', candidate_images)
	np.save('ucf_labels.npy', candidate_labels)
	np.save('ucf_llc.npy', candidate_llc)
	np.save('ucf_targets.npy', candidate_targets)



	


