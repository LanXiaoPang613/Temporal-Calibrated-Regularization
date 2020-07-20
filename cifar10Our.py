import argparse
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# import models
import numpy as np
from PIL import Image
import os
import os.path
import sys
import resnet
import pre_ResNet
import CNN_9layer
import torch.nn.functional as F

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from train_val_dataloader import load_data_sym10, load_data_asym10

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='H-P', help='initial learning rate')
parser.add_argument('--lr2', '--learning-rate2', default=0.2, type=float,
                    metavar='H-P', help='initial learning rate of stage3')
parser.add_argument('--gamma', default=1.1, type=float,
                    metavar='H-P', help='the coefficient of Compatibility Loss')
parser.add_argument('--beta', default=0.1, type=float,
                    metavar='H-P', help='the coefficient of Entropy Loss')
parser.add_argument('--lambda1', default=600, type=int,
                    metavar='H-P', help='the value of lambda')
parser.add_argument('--stage1', default=80, type=int,
                    metavar='H-P', help='number of epochs utill stage1')
parser.add_argument('--stage2', default=150, type=int,
                    metavar='H-P', help='number of epochs utill stage2')
parser.add_argument('--epochs', default=270, type=int, metavar='H-P',
                    help='number of total epochs to run')
parser.add_argument('--datanum', default=45000, type=int,
                    metavar='H-P', help='number of train dataset samples')
parser.add_argument('--classnum', default=10, type=int,
                    metavar='H-P', help='number of train dataset classes')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', dest='gpu', default='0', type=str,
                    help='select gpu')
parser.add_argument('--dir', dest='dir', default='result', type=str, metavar='PATH',
                    help='model dir')

best_prec2 = 0
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if epoch < args.stage1:
        # lr = args.lr
        lr = 0.1
    elif epoch < 120:
        # current_epoch = (epoch + 1 - args.stage2) % args.freq
        # # print('current epoch ',epoch, current_epoch)
        # s_t = (1+(epoch-69-1) % args.freq) / args.freq
        # # print(s_t)
        # r_t = (1-s_t)*0.01 + s_t * 0.001
        # lr = r_t
        lr = 0.01
    else:
        lr = 0.001
    print('epoch %d, learning rate %f'%(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def LogSoftmax(factor, inputs):
    return F.softmax(factor, 1) * F.log_softmax(inputs, 1)

def SumLogSoftmax(factor, inputs):
    log_loss = F.kl_div(F.log_softmax(inputs, dim=1),F.softmax(factor, dim=1),reduce=False)#-F.softmax(factor, 1) * F.log_softmax(inputs, 1)
    h = torch.mean(log_loss, 1)
    loss_log = torch.mean(log_loss)
    return loss_log, h

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

LR_RATE = 0.1
H=0
if __name__ == '__main__':
    aaa = [0.2, 0.4, 0.6, 0.8, 0.1]#[0.5, 0.4, 0.3, 0.2, 0.1]
    K = aaa[H]
    txtfile = 'noise_detect_tcr ' + str(aaa[H]) + ".txt"
    model = pre_ResNet.ResNet18(10)
    # model = torch.load("backmodel1.pth")
    model = model.to(device)
    # if os.path.exists('ourInit_model.pt')==False:
    #     torch.save(model.state_dict(), 'ourInit_model.pt')
    #     print('save init model')
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), LR_RATE,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True
    isPrint = True
    train_loader, valloader, _ = load_data_sym10(noise_rate=aaa[H])
    new_y1 = np.zeros([50000, 1])
    new_y = torch.zeros([50000, 10])
    loss_record = np.zeros([50000, 1])
    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.stage2):
        adjust_learning_rate(optimizer, epoch)
        model.train()
        losses = 0.
        total = 0
        for data in train_loader:
            input, target, index, one_hot, _ = data
            index = index.numpy()
            target_var = target.to(device)
            input_var = input.to(device)
            one_hot = one_hot.to(device)
            output = model(input_var)
            temp_zi = torch.softmax(output, 1).detach().cpu()
            if epoch == 0:
                yi = one_hot
            else:
                yi = args.beta*one_hot + (1.-args.beta)*(new_y[index].to(device))
            loss, reduce_loss = SumLogSoftmax(yi, output)
            new_y1[index, :] = reduce_loss.detach().cpu().numpy().reshape(-1, 1)
            if epoch >= args.stage1:
                # squeeze
                squeeze_1 = temp_zi**args.gamma
                squeeze_2 = torch.sum(temp_zi**args.gamma, 1)
                for ii in range(len(index)):
                    new_y[index[ii]] = (squeeze_1[ii]/squeeze_2[ii]).numpy()
            else:
                new_y[index] = temp_zi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss
            total += 1
        mean_loss = np.mean(new_y1, axis=0)
        new_y1 = new_y1 - mean_loss
        loss_record[:] += new_y1[:]
        torch.cuda.empty_cache()
        print('epoch{%d}/{320}, loss:%f' % (epoch, losses / total))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                input, target = data
                target = target.to(device)
                input = input.to(device)
                outputs = model(input)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += torch.sum(predicted == target)
            pre_val = correct.double() / total
            print('model_prec%.5f, best_prec%.5f' % (pre_val, best_prec2))
            best_prec2 = max(pre_val, best_prec2)
            with open(txtfile, "a") as myfile:
                myfile.write(str(int(epoch)) + ' ' + str(float(pre_val.data)) + ' ' +
                             str(float(best_prec2.data)) + ' '+"\n")
                # myfile.write(str(float(correct_num)) + "\n")
    # sort the loss and remove noisy index
    torch.cuda.empty_cache()
    index_noisy = np.argsort(-loss_record, axis=0)
    # i_temp = np.argsort(new_y, axis=0)
    num_select = int(K * len(loss_record))
    select_index = index_noisy[:num_select]
    np.save('symmetric_'+str(K), select_index)
    train_loader.dataset.update_noisy_label(select_index)
    # print('len of updated sets',len(train_loader.dataset))
    # model.load_state_dict(torch.load('ourInit_model.pt'))
    print('re-init successfully')
    for epoch in range(args.stage2, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        model.train()
        losses = 0.
        total = 0
        for data in train_loader:
            input, target, index, _, _ = data
            index = index.numpy()
            target_var = target.to(device)
            input_var = input.to(device)
            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss
            total += target.size(0)
        torch.cuda.empty_cache()
        print('epoch{%d}/{320}, loss:%f' % (epoch, losses / total))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                input, target = data
                target = target.to(device)
                input = input.to(device)
                outputs = model(input)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += torch.sum(predicted == target)
            pre_val = correct.double() / total
            print('model_prec%.5f, best_prec%.5f' % (pre_val, best_prec2))
            best_prec2 = max(pre_val, best_prec2)
            with open(txtfile, "a") as myfile:
                myfile.write(str(int(epoch)) + ' ' + str(float(pre_val.data)) + ' ' +
                             str(float(best_prec2.data)) + ' '+"\n")
                # myfile.write(str(float(correct_num)) + "\n")
