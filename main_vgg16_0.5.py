import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

import vgg4 as vgg
from tensorboardX import SummaryWriter
import numpy as np
import random

from MyDataset import CLSDataPrepare,classifier_collate
from MyAugmentations import TrainAugmentation,TestAugmentation
from names import class_names

import datetime

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))
ratio = '0.5'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--arch', '-a', metavar='ARCH', default='orn_align2d_vgg16',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')

parser.add_argument('--pretrained', default='./pretrained_model/IOR4-VGG16_on_ImageNet_checkpoint_89', type=str, metavar='PATH', help='use pre-trained model')
# parser.add_argument('--pretrained', default=None, type=str, metavar='PATH', help='use pre-trained model')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')


parser.add_argument('-e', '--evaluate', default=False,
                    help='evaluate model on validation set')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')


parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_dir_' + ratio, type=str)


time_now = datetime.datetime.now()
time_str = datetime.datetime.strftime(time_now,'%Y%m%d_%H_%M_%S')

time_dir = './'+time_str
os.makedirs(time_dir)
logdir = time_dir + '/log_' + ratio
save_dir = time_dir + '/save_' + ratio

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
TFwriter = SummaryWriter(logdir)

data_basepath = './datasets/'
dataset_name = 'UCMerced_LandUse/Images'
img_paths = data_basepath + dataset_name + '/' 
fd = open(logdir+'/matrix.csv','w') # confuion matrix
dataset_mean = [0.45050276, 0.49005175, 0.48422758]
dataset_std = [0.19583832, 0.2020706, 0.2180214]
adjust_lr_epoch = 50 # adjust learning rate each 50 epoches

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = vgg.__dict__[args.arch]()

    # model.features = torch.nn.DataParallel(model.features)
    if args.pretrained is not None:
        print("=> loading pretrained model '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint)
        model.classifier.add_module('7', nn.ReLU(inplace=True))
        model.classifier.add_module('8', nn.Dropout())
        model.classifier.add_module('9', nn.Linear(1000, len(class_names)))
        # for para in model.parameters():
        #   para.requires_grad = False
        # for para in model.classifier[9].parameters():
        #   para.requires_grad = True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print ("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
        else:
            print ("=> no checkpoint found at '{}'".format(args.resume))

    model.cuda()
    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        CLSDataPrepare(txt_path= img_paths + 'trainval' + ratio + '.txt',
                          img_transform=TrainAugmentation(size=224, _mean = dataset_mean, _std=dataset_std)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn = classifier_collate)

    val_loader = torch.utils.data.DataLoader(
        CLSDataPrepare(txt_path=img_paths + 'val' + ratio + '.txt',
                          img_transform=TestAugmentation(size=224, _mean = dataset_mean, _std=dataset_std)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn = classifier_collate)

    test_loader = torch.utils.data.DataLoader(
        CLSDataPrepare(txt_path=img_paths + 'test' + ratio + '.txt',
                          img_transform=TestAugmentation(size=224, _mean = dataset_mean, _std=dataset_std)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn = classifier_collate)    

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(model.classifier[9].parameters(), args.lr,
    #                           momentum=args.momentum,
    #                           weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        prec1, single = validate(test_loader, model, criterion, len(class_names))
        print(prec1)
        return

    best_accuracy = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, TFwriter)

        # evaluate on validation set
        # val_loss = validate_loss(val_loader, model, criterion, len(class_names))
        # TFwriter.add_scalar('#val_loss', val_loss, epoch)

        prec1, Confusion_Matrix, test_loss= validate(val_loader, model, criterion, len(class_names))
        TFwriter.add_scalar('#test_loss', test_loss, epoch)
        TFwriter.add_scalar('#accuracy', prec1, epoch)

        print('after %d epochs,accuracy = %f, test_loss = %f'%(epoch, prec1, test_loss))

        # remember best prec@1 and save checkpoint
        # if prec1 > best_accuracy:
        #     best_accuracy = prec1
        #     torch.save(model.state_dict(),
        #                os.path.join(args.save_dir,
        #                             'checkpoint_{}_{best_prec1:.3f}.pth'.format(epoch, best_prec1 = best_accuracy)))
        if prec1 > best_accuracy:
            best_accuracy = prec1
            fd = open(logdir+'/matrix.csv','w')
            sumline = np.sum(Confusion_Matrix, axis=1)  
            for (i,line) in enumerate(Confusion_Matrix):
                for col in line:
                    value = col/sumline[i]
                    if value == 0:
                        fd.write(',')
                    else:
                        fd.write(str(value)+',')
                fd.write('\r\n') 
            fd.close()


def train(train_loader, model, criterion, optimizer, epoch,TFwriter):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        step = i+len(train_loader)*epoch+1
        # print(step)

        if step % args.print_freq == 0:
            TFwriter.add_scalar('#loss', loss.data.cpu().numpy(), step)
            print('step %d: loss = %f'%(step, loss.data.cpu().numpy()))


def validate(val_loader, model, criterion, num_of_classes):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    correct_single_num = np.zeros(num_of_classes)
    target_single_num = np.zeros(num_of_classes)

    end = time.time()
    Confusion_Matrix = np.zeros([num_of_classes, num_of_classes])
    for i, (input, target) in enumerate(val_loader):
        # # RandomRotation
        # N_90 = int(random.random()/0.25)
        # if N_90 != 0:
        #     input0 = np.rot90(input.numpy(), N_90, (2,3))
        #     input = torch.from_numpy(input0.copy())

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)


        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
##############################################################
        topk = (1,)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.data.topk(maxk, 1, True, True)
        pred = pred.t()
        target_all = target.view(1, -1).expand_as(pred)

        y = target_all.view(-1)
        x = pred.view(-1)
        for (ii,yy) in enumerate(y):
            Confusion_Matrix[yy][x[ii]] += 1


################################################################
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, Confusion_Matrix, losses.avg


def validate_loss(val_loader, model, criterion, num_of_classes):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    correct_single_num = np.zeros(num_of_classes)
    target_single_num = np.zeros(num_of_classes)

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)


        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()
        losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return losses.avg

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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // adjust_lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target_all = target.view(1, -1).expand_as(pred)
    # all
    correct = pred.eq(target_all)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
