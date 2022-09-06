'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import os
import sys
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
#!pip install ptflops
from ptflops import get_model_complexity_info
import os
import argparse

from convnet_aig import *
#!pip install visdom
from visdom import Visdom
import numpy as np
import gc


expname='give_me_a_name_cifar'
lrfact=1
lossfact=2
target=0.6
batch_size=256
epochs=350
start_epoch=1
lr=0.1
momentum=0.9
weight_decay=5e-4
seed=1
log_interval=20
resume=False
test=False
visdom = False
print_freq=25

best_prec1 = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    global best_prec1,expname,lrfact,lossfact,target,batch_size,epochs
    global start_epoch, lr, momentum, weight_decay, seed, log_interval, resume, test, visdom, print_freq
    #torch.cuda.manual_seed(seed)

    if visdom:
        global plotter 
        plotter = VisdomLinePlotter(env_name=expname)
    
    # Data loading code
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    kwargs = {'num_workers': 2}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=False, **kwargs)
    
    model = ResNet110_cifar(nclass=10)
    model.to(device)


    # optionally resume from a checkpoint
    if resume:
        latest_checkpoint = os.path.join(resume, 'checkpoint.pth.tar')
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))


    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if 'fc2' in name],
                            'lr': lrfact * lr, 'weight_decay': weight_decay},
                            {'params': [param for name, param in model.named_parameters() if 'fc2' not in name],
                            'lr': lr, 'weight_decay': weight_decay}], 
                            momentum=momentum)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    #macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
    #                                       print_per_layer_stat=True, verbose=True)
    #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    if test:
        test_acc = validate(val_loader, model, criterion, 350)
        sys.exit()

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""

    # Temperature of Gumble Softmax 
    # We simply keep it fixed
    temp = 1

    # switch to train mode
    model.train()

    end = time.time()

    
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        # compute output
        output, activation_rates = model(input, temperature=temp)

        # classification loss
        loss_classify = criterion(output, target)

        # target rate loss
        acts = 0
        acts_plot = 0
        for act in activation_rates:
            acts_plot += torch.mean(act)
            acts += torch.pow(target - torch.mean(act), 2)

        # this is important when using data DataParallel
        acts_plot = torch.mean(acts_plot / len(activation_rates))
        acts = torch.mean(acts / len(activation_rates))

        act_loss = lossfact * acts
        loss = loss_classify  + act_loss

        # measure accuracy
        prec1 = accuracy(output, target, topk=(1,))[0]
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        time_total = time.time() - end
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {time_total:.3f} ({time_total:.3f})\t'
                  'Loss {loss:.4f} ({loss:.4f}) c({lossc:.4f}) a({lossa:.4f})\t'
                  'Prec@1 {prec1:.3f} ({prec1:.3f})\t'
                  'Activations: {act:.3f} ({act:.3f})'.format(
                      epoch, i, len(train_loader), time_total=time_total,
                      loss=loss, lossa=act_loss, lossc=loss_classify, prec1=prec1, act=acts_plot))

    # log values to visdom
    if visdom:
        plotter.plot('top1', 'train', epoch, top1.avg)
        plotter.plot('loss', 'train', epoch, losses.avg)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    accumulator = ActivationAccum(epoch=epoch)
    
    # Temperature of Gumble Softmax 
    # We simply keep it fixed
    temp = 1

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target_var = target.to(device)
        input_var = input.to(device)
        print('start test')

        # compute output
        output, activation_rates = model(input_var, temperature=temp)
        print('finished test')

        # classification loss
        loss = criterion(output.detach(), target_var)

        # target rate loss
        acts = 0
        for act in activation_rates:
            acts += torch.mean(act) 
        # this is important when using data DataParallel
        acts = torch.mean(acts / len(activation_rates))

        # accumulate statistics over eval set
        accumulator.accumulate(activation_rates, target_var.detach())

        # measure accuracy and record loss
        prec1 = accuracy(output.detach(), target_var.detach(), topk=(1,))[0]

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time:.3f} ({batch_time:.3f})\t'
                  'Loss {loss:.4f} ({loss:.4f})\t'
                  'Prec@1 {top1:.3f} ({top1:.3f})\t'
                  'Activations: {act:.3f} ({act:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=loss,
                      top1=prec1, act=acts))
    activ_output = accumulator.getoutput()

    print('gate activation rates:')
    print(activ_output[0])

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    if visdom:
        plotter.plot('top1', 'test', epoch, top1.avg)
        plotter.plot('loss', 'test', epoch, losses.avg)
        for gate in activ_output[0]:
            plotter.plot('gates', '{}'.format(gate), epoch, activ_output[0][gate])
        
        if epoch % 25 == 0:
            for category in activ_output[1]:
                plotter.plot('classes', '{}'.format(category), epoch, activ_output[1][category])

            heatmap = activ_output[2]
            means = np.mean(heatmap, axis=0)
            stds = np.std(heatmap, axis=0)
            normalized_stds = np.array(stds / (means + 1e-10)).squeeze()
            
            plotter.plot_heatmap(activ_output[2], epoch)
            for counter in range(len(normalized_stds)):
                plotter.plot('activations{}'.format(epoch), 'activations', counter, normalized_stds[counter])
            for counter in range(len(means)):
                plotter.plot('opening_rate{}'.format(epoch), 'opening_rate', counter, means[counter])
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(expname) + 'model_best.pth.tar')

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name], name=split_name)
    def plot_heatmap(self, map, epoch):
        self.viz.heatmap(X = map, 
                         env=self.env,
                         opts=dict(
                                    title='activations {}'.format(epoch),
                                    xlabel='modules',
                                    ylabel='classes'
                                ))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    global lr
    factor = lrfact
    if epoch >= 150:
        lr = 0.1 * lr
    if epoch >= 250:
        lr = 0.1 * lr
    if visdom:
        plotter.plot('learning_rate', 'train', epoch, lr)
    optimizer.param_groups[0]['lr'] = factor * lr
    optimizer.param_groups[1]['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()   