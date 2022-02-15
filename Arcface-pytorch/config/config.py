import os
from os import getcwd

wd = getcwd()

class Config(object):
    env = 'default'
    backbone = 'resnet50'
    classify = 'softmax'
    num_classes = 500
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    #train.py
    train_root = wd+'/data'
    train_list = wd+'/data/cls_train.txt'

    val_list = '/data/Datasets/webface/val_data_13938.txt'


    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = wd+'/data/lfw-align-128'
    lfw_test_list = wd+'/data/lfw_test_pair.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/resnet50_tuned.pth'#'checkpoints/resnet18_110.pth'
    save_interval = 10

    train_batch_size = 16  # batch size
    test_batch_size = 60

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    #num_workers = 4  # how many workers for loading data
    #FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
    #https://blog.csdn.net/Warmth_Dream/article/details/108336455
    num_workers = 0
    print_freq = 100  # print info every N batch #100

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 25 #50
    lr = 1e-1  # initial learning rate
    lr_step = 10 # 10
    lr_decay = 0.99  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
