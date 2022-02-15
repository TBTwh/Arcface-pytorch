import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys

import os
from os import getcwd

wd = getcwd()



class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(1, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        #data_list_file 按照每行存储
        #经.readlines()之后，被识别成：['line1\n','line2\n','line3']

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        # path.join()连接成['root'+'line1\n','root'+'line2\n','root'+'line3']
        #在每行中选择img[:-1]可以除去每个\n,得到['root'+'line1','root'+'line2','root'+'line3']
        # imgs[i] = "img_path lable"
        self.imgs = np.random.permutation(imgs)
        #随机排列一个序列，返回一个排列的序列。

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        normalize = T.Normalize(mean=[0.5], std=[0.5])
        #把0-1变换到(-1,1)，把0-1变换到(-1,1)

        if self.phase == 'train':
            #torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
                #在一个随机的位置进行裁剪
                T.RandomHorizontalFlip(),
                #以0.5的概率水平翻转给定的PIL图像
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                #在图片的中间区域进行裁剪
                T.ToTensor(),
                normalize
            ])


    # dataset必须继承自torch.utils.data.Dataset,内部要实现两个函数
    # 一个是__lent__用来获取整个数据集的大小
    # 一个是__getitem__用来从数据集中得到一个数据片段item

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        #str.split(str="", num=string.count(str)).
        #str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
        #num -- 分割次数。默认为 -1, 即分隔所有。

        img_path = splits[0]
        data = Image.open(img_path)
        data = data.convert('L')
        #为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
        #转换公式：L = R * 299/1000 + G * 587/1000+ B * 114/1000。

        data = self.transforms(data)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(root='/data/Datasets/fv/dataset_v1.1/dataset_mix_aligned_v1.1',
                      data_list_file='/data/Datasets/fv/dataset_v1.1/mix_20w.txt',
                      phase='test',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # make_grid的作用是将若干幅图像拼成一幅图像
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)