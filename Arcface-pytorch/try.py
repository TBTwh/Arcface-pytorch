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
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


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

    #print('{} train iters per epoch:'.format(len(trainloader)))
    #149 train iters per epoch

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    #print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)
    # scheduler讲解
    # https://zhuanlan.zhihu.com/p/136902153

    start = time.time()
    for i in range(opt.max_epoch):
        

        model.train()
        '''
        # model.train()的作用是启用 Batch Normalization 和 Dropout。
        # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()
        # model.train()是保证BN层能够用到每一批数据的均值和方差
        # 对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
        # https://zhuanlan.zhihu.com/p/357075502
        '''
        for ii, data in enumerate(trainloader):
            optimizer.zero_grad()
            # zero_grad()操作应该放在forward之前的，流程应该是zero_grad()+forward + backward + optimize
            # https://zhuanlan.zhihu.com/p/342764133
            
            data_input, label = data           
            data_input = data_input.to(device)
            label = label.to(device).long()        
            #print('data_input:{}\n'.format(data_input.shape))
            #print('label:{}\n'.format(label.shape))
            # data_input.shape:torch.Size([16, 1, 128, 128])
            # label.shape:torch.Size([16])

            feature = model(data_input)
            #print('feature:{}\n'.format(feature.shape))
            # feature.shape:torch.Size([16, 512])

            # metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
            output = metric_fc(feature, label)
            #print('output.shape:{}\n'.format(output.shape))
            #print('output {}\n\n\n'.format(output))
            #output.shape:torch.Size([16, 500])

            loss = criterion(output, label)
            print('loss {}\n'.format(loss))
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            #scheduler用于调整学习率
            
            break
            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        break

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
        acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        if opt.display:
            visualizer.display_current_results(iters, acc, name='test_acc')
