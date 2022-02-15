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





if __name__ == '__main__':
    
    opt = Config()
    device = torch.device("cuda")

    #model
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()
    
    model = DataParallel(model)#需要放在to device前否则参数读取出错
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path),False)
    model.to(torch.device("cuda"))
    
    model.eval()

    #dataset
    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    #metic
    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)


    #predict
    for ii, data in enumerate(trainloader):
        data_input, label = data
        data_input = data_input.to(device)
        label = label.to(device).long()
        feature = model(data_input)
        output = metric_fc(feature, label)
        print(' lable.shape = {}\n label = {}\n\n\ output.shape = {}\n output = {}\n'.format(label.shape,label,output.shape,output))
        
        
        prob = F.softmax(output,dim=1)
        print('prob.shape = {}\n prob = {}\n\n\ '.format(prob.shape,prob))
        result = torch.argmax(prob, dim=1)
        print('result.shape = {}\n result = {}\n\n\ '.format(result.shape,result))
        break