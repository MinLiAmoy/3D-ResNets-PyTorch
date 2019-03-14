import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy
from attacks import fgsm


def attack(data_loader, model, criterion, opt, logger):

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    losses_adv = AverageMeter()
    accuracies_adv = AverageMeter()

    epsilons = [0, .05, .1, .15, .2, .25, .3]

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        inputs.requires_grad = True
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        model.zero_grad()
        loss.backward()
        inputs_grad = inputs.grad.data
        inputs_adv = fgsm.attack(inputs, epsilons[6], inputs_grad)
        outputs = model(inputs_adv)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses_adv.update(loss.data.item(), inputs_adv.size(0))
        accuracies_adv.update(acc, inputs_adv.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Batchs:[{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'Acc_adv {acc_adv.val:.3f} ({acc_adv.avg:.3f})'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  acc=accuracies,
                  acc_adv=accuracies_adv))

    logger.log({'loss': losses.avg, 'acc': accuracies.avg, 'loss_adv': losses_adv.avg,
                'acc_adv': accuracies_adv.avg})

