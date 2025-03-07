from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import math
from matplotlib.colors import Normalize
from torchstat import stat

from vgg import vgg
import shutil

parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--percent', type=float, default=0.1,
                    help='scale sparse rate (default: 0.1)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--a', type=float, default=0.1,
                    help='Scale factor termination threshold (default: 0.1)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='refine from prune model')
parser.add_argument('--grow', default='', type=str, metavar='PATH',
                    help='The path of growing')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 3)')#
parser.add_argument('--cuttingtime', type=int, default=3, metavar='N',
                    help='The time to pruning')
parser.add_argument('--growingtime', type=int, default=3, metavar='N',
                    help='The time to growing')
parser.add_argument('--net', type=int, default=5, metavar='N',
                    help='Choose what net you want to test')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')#
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

def Pre_pruning(cfg,cfg_mask,cfg1,mask1):
    flag=0
    for i in cfg1:
        if i != 'M':
            cfg.append(int(torch.sum(mask1[0+flag:i+flag])))
            flag+=i
        else:
            cfg.append('M')
    print(cfg)
    flag1=0
    for i in cfg1:
        if i != 'M':
            cfg_mask.append(mask1[0+flag1:i+flag1])
            flag1+=i

def compute_normalized_dot_products(list1, list2):
    """
    Calculate the dot product sum of the corresponding tensors in two lists and normalize them.

    list1 (list):
    list2 (list): Reverse order

    return:
    list: Calculate the dot product of each tensor pair and normalize the resulting list
    """
    assert len(list1) == len(list2), "The length of two lists must be the same"
    
    results = []
    Index=[]

    for a, b in zip(list1, reversed(list2)):

        a_mean = torch.mean(a, dim=0)
        b_mean = b.mean(dim=0) 
        b_sum = torch.sum(b_mean, dim=0)
        
        # unfold,im2col
        cols = F.unfold(b_sum.unsqueeze(0).unsqueeze(0), kernel_size=(3, 3), padding=1, stride=1)
        cols = cols.squeeze(0).squeeze(0)

        d_input_cols = cols*0.01
        
        x_grad= col2im(d_input_cols, a_mean.shape[1:], 3, 3, stride=1, pad=1)
        
        normalized_result = torch.sum(x_grad*a_mean,dim=(1, 2))
 
        min_index = torch.argmin(normalized_result)
        
        Index.append(min_index)
        results.append(normalized_result[min_index])
    
    return results,Index

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=1):
    H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = torch.zeros((H + 2 * pad, W + 2 * pad)).cuda()

    col = col.reshape(filter_h * filter_w, out_h * out_w)

    for y in range(out_h):
        for x in range(out_w):
            img[y * stride:y * stride + filter_h, x * stride:x * stride + filter_w] += col[:, y * out_w + x].reshape(filter_h, filter_w)
    
    # Remove the filling part
    return img[pad:H + pad, pad:W + pad]

def Process_of_pruning(cfg_mask,model,newmodel):
    layer_id_in_cfg = 0 
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg] 
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1 
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask): # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d): 
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d} Out shape:{:d}'.format(idx0.size, idx1.size))
            w = m0.weight.data[:, idx0, :, :].clone()
            w = w[idx1, :, :, :].clone()
            m1.weight.data = w.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            
def update_layer_element(cfg_grow, p):
    index = 0
    flag = 0
    list1=[1, 1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8]
    result = [p * list1 for p, list1 in zip(p, list1)]
    for i in cfg_grow:
        if i != 'M':
            cfg_grow[index] += result[flag]
            flag += 1
        index += 1
    print(cfg_grow)

def Process_of_growing(model,newmodel,INDEX):
    layer_id=0
    index=INDEX
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d): 
            idx1 = m0.weight.data.shape[0]
            m1.weight.data[0:idx1] = m0.weight.data.clone()
            m1.bias.data[0:idx1] = m0.bias.data.clone()
            m1.weight.data[idx1:] = m0.weight.data[index[layer_id]]
            m1.bias.data[idx1:] = m0.bias.data[index[layer_id]]
            m1.running_mean[0:idx1] = m0.running_mean.clone()
            m1.running_var[0:idx1] = m0.running_var.clone()
            layer_id += 1
        elif isinstance(m0, nn.Conv2d):
            idx0 = m0.weight.data.shape[1]
            idx1 = m0.weight.data.shape[0]
            # print('In shape: {:d} Out shape:{:d}'.format(idx0.size, idx1.size))
            m1.weight.data[0:idx1, 0:idx0, :, :] = m0.weight.data.clone()
           
            m1.weight.data[idx1:, 0:idx0, :, :] = m0.weight.data[index[layer_id]]

            m1.weight.data[:, idx0:, :, :] = 0.01
        elif isinstance(m0, nn.Linear):
            idx0 = m0.weight.data.shape[1]
            m1.weight.data[:, 0:idx0] = m0.weight.data.clone()
            m1.weight.data[:, idx0:] = 0.01
            
def save_plot_line(x_data, y_data, title, x_label, y_label, legend_label, filename, col):

    plt.plot(x_data, y_data, label=legend_label, linestyle='-', color=col, linewidth=2, markersize=10)

    plt.title(title, fontdict={'family': 'serif', 'size': 18, 'color': 'black', 'weight': 'bold'})
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(loc='best')
    plt.grid(linestyle='--', alpha=0.5, linewidth=2)

    plt.savefig(filename, dpi=300)

    plt.show()
    plt.clf()
def heat_map(bn, filename, ep, total):
    """
        Draw a heatmap of the scaling factor of the bn layer with respect to the number of epochs and its own index, generally used to observe the changes in the factor after the training is completed
        
    """
    plt.imshow(bn, cmap='viridis', interpolation='nearest', aspect='auto', extent=(0, ep, 0, total), vmin=0, vmax=1)

    plt.xlabel('Epochs')
    plt.ylabel('The index of BN_gamma')

    plt.savefig(filename, dpi=300)

    plt.show()

def cumcount(X):
    
    sum1=sum(X)
    
    return sum1

def updateBN():#BN update
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  #L1

def count_parameter(model):
    totally = 0  # Count the parameter quantities of all BN layers
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            totally += m.weight.data.shape[0]
    return totally

def get_of_BN(epoch,bn,model):
    """
    Obtain the scaling factors of all bn layers in the current epoch and model
    """
    index = 0
    for m in model.modules():
        # copy bn
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[epoch][index:(index + size)] = m.weight.data.abs().clone()
            index += size
            
def getplot_of_BN(bn,model):
    """
    for heat map
    """
    index = 0
    for m in model.modules():
       
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
           
            index += size
            
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data, use_hooks=True)
        loss = F.cross_entropy(output, target)
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data, use_hooks=False)
            test_loss += F.cross_entropy(output, target,reduction='sum').data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    
    defaultcfgg = {
    5 : [96, 'M', 256, 'M', 384, 384, 256, 'M'],#Alexnet
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    21 : [59, 55, 'M', 134, 142, 'M', 224, 204, 232, 252, 'M', 432, 360, 464, 144, 'M', 152, 176, 192, 456],
    20 : [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 64, 64, 64, 64],
    }#vgg family and Alexnet
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    remember_time=20
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    datatrain = datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),  
                             transforms.RandomCrop(32), 
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    datatest = datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
   
    train_loader = torch.utils.data.DataLoader(datatrain, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datatest,batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = vgg()
    
    if args.cuda:
        model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    ave = [x for x in range(1, args.epochs+1)]
    prec1 = [x for x in range(1, args.epochs + 1)]
    Heat=[]
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs * 0.6, args.epochs * 0.7] or epoch in [args.epochs * 0.8, args.epochs * 0.9]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        if args.refine and (epoch) == args.cuttingtime and epoch != 0:
            
            checkpoint = torch.load(args.refine)
            model = vgg(cfg=checkpoint['cfg2'])
            model.load_state_dict(checkpoint['state_dict'])

            print(model)

            train_loader = torch.utils.data.DataLoader(datatrain, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(datatest, batch_size=args.test_batch_size, shuffle=True, **kwargs)
            
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            
            if args.cuda:
                 model.cuda()
                    
        if args.grow and (epoch) <= args.growingtime and epoch != 0:
            checkpoint = torch.load(args.grow)
            model = vgg(cfg=checkpoint['cfg'])
            model.load_state_dict(checkpoint['state_dict'])

            train_loader = torch.utils.data.DataLoader(datatrain, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(datatest, batch_size=args.test_batch_size, shuffle=True, **kwargs)
            
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            
            if args.cuda:
                 model.cuda()
    
        train(epoch)
        
        bn_total = count_parameter(model)
        bn_plot = torch.zeros(bn_total)
        getplot_of_BN(bn_plot, model)
        
        Heat.append(bn_plot)
        
        if (epoch) == args.growingtime:
            
            total = count_parameter(model)
   
            bn = torch.zeros(args.cuttingtime-args.growingtime, total)
    
        if (epoch) >= args.growingtime and (epoch+1) <= args.cuttingtime:

            get_of_BN(epoch-args.growingtime, bn, model)
            
        flattened_x = torch.mean(model.flattened_x, dim=0)
        y_grad = torch.mean(model.y_grad, dim=0)
    
        y_grad_sum = torch.sum(y_grad)*0.01
       
        comparison = y_grad_sum * flattened_x

        min_index = torch.argmin(comparison)
        
        p1,INDEX=compute_normalized_dot_products(model.relu_outputs, model.conv_gradients)
        p1.append(comparison[min_index])     
        INDEX.append(min_index)
       
        p2 = [tensor.item() for tensor in p1]
        
        threshold = np.percentile(p2, 25)

        p3 = [0 if x > threshold else 1 for x in p2]
            
        if (epoch+1) == args.cuttingtime :
            
            bncum = torch.zeros(total)
            for i in range(0,total):
                bncum[i]= cumcount(bn[0:args.cuttingtime-args.growingtime,i])
            yy, ii = torch.sort(bncum)
            thre_index1 = int(total * args.percent) 

            thre1 = yy[thre_index1] 
            mask1 = bncum.gt(thre1).float().cuda()

            cfg2 = []
            cfg_mask = []
            
            Pre_pruning(cfg2,cfg_mask,cfg,mask1)
 
            newmodel = vgg(cfg=cfg2)
            newmodel.cuda()

            Process_of_pruning(cfg_mask, model, newmodel)

            torch.save({'cfg2': cfg2, 'state_dict': newmodel.state_dict()}, args.refine)
        if (epoch + 1) <= args.growingtime :
            
            if epoch==0:
                cfg = defaultcfgg[args.net]
            else:
                checkpoint = torch.load(args.grow)
                cfg=checkpoint['cfg']
            
            update_layer_element(cfg, p3)

            newmodel = vgg(cfg=cfg)
            newmodel.cuda()

            Process_of_growing(model, newmodel,INDEX)

            torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.grow)
        ave[epoch], prec1[epoch] = test(model)

    data=Heat
    normalized_data = []
    for tensor in data:
        arr = tensor.numpy()
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val != min_val:  # avoid ÷ 0
            normalized_data.append((arr - min_val) / (max_val - min_val))
        else:
            normalized_data.append(np.zeros_like(arr))
    data=normalized_data
    max_len = max(len(sublist) for sublist in data)

    data_filled = np.full((max_len, len(data)), np.nan)

    for i, sublist in enumerate(data):
        data_filled[max_len - len(sublist):, i] = sublist

    plt.figure(figsize=(12, 6))
    plt.imshow(data_filled, cmap='YlGn', interpolation='nearest', aspect='auto')

    plt.colorbar()
    
    xticks = np.arange(0, len(data), 100)
    plt.xticks(ticks=xticks, labels=xticks)
    plt.yticks([])
   
    plt.xlabel('List Index') 
    plt.ylabel('Dimension Index') 
    plt.title('Heatmap of Nested Tensors')

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

    plt.savefig('heatmap_tensor_fixed_500ticks.png', dpi=300)

    plt.show()
    x1= [x for x in range(1, args.epochs+1)]
    x2 = [x for x in range(1, args.epochs + 1)]
    
    save_plot_line(x1, ave, 'Test Set Average loss - Epochs Plot', 'Epochs', 'Test Set Average loss', 'Test Set Average loss', '1.png', 'blue')
    save_plot_line(x2, prec1, 'Accu -Epochs Plot', 'Epochs', 'Accuracy', 'Accuracy', '2.png', 'red')
    
