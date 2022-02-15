from __future__ import print_function
import os
from data import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
from utils import Visualizer, view_model
import torch
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_new' + str(iter_cnt) + '.pth')
    print('save_name = {}'.format(save_name))
    torch.save(model.state_dict(), save_name)
    return save_name

'''
一个python文件通常有两种使用方法:
    第一是作为脚本直接执行，
    第二是 import 到其他的 python 脚本中被调用（模块重用）执行。
因此 if __name__ == 'main': 的作用就是控制这两种情况执行代码的过程，
在 if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行
而 import 到其他脚本中是不会被执行的。

'''

if __name__ == '__main__':

    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        #model = resnet50()
        model = resnet_face50()
    elif opt.backbone == 'resnet101':
        model = resnet_face101()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':                       #s=30
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=64, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    #print(model)

    
    # 初始训练时候加载已有模型
    model = DataParallel(model)    
    model.load_state_dict(torch.load(opt.test_model_path),False)
    model.to(device)
    '''
    model.to(device)
    model = DataParallel(model) 
    '''
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)
    # scheduler讲解
    # https://zhuanlan.zhihu.com/p/136902153

    start = time.time()
    for i in range(opt.max_epoch):
        

        model.train()
        metric_fc.train()
        '''
        # model.train()的作用是启用 Batch Normalization 和 Dropout。
        # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()
        # model.train()是保证BN层能够用到每一批数据的均值和方差
        # 对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
        # https://zhuanlan.zhihu.com/p/357075502
        '''
        for ii, data in enumerate(trainloader):
            #optimizer.zero_grad()
            # zero_grad()操作应该放在forward之前的，流程应该是zero_grad()+forward + backward + optimize
            # https://zhuanlan.zhihu.com/p/342764133
            
            
            data_input, label = data
            data_input = data_input.to(device)  # data_input.shape:torch.Size([16, 1, 128, 128])
            label = label.to(device).long() # label.shape:torch.Size([16])                    
            feature = model(data_input)     # feature.shape:torch.Size([16, 512])
            output = metric_fc(feature, label)  #output.shape:torch.Size([16, 500])
            loss = criterion(output, label)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            #scheduler用于调整学习率
            

            iters = i * len(trainloader) + ii
            
            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                #print('output = {}  lable = {}\n'.format(output,label))
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()
            
        
        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
        metric_fc.eval()
        acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        if opt.display:
            visualizer.display_current_results(iters, acc, name='test_acc')