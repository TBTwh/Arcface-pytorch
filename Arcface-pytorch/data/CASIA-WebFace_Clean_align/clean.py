# 1.用于清理CASIA-WebFace中低质量的图片
# 2.用于建立指定根目录下的CASIA-WebFace数据列表
# 2022-2-15
import os
from os import getcwd
import os.path as osp
from imutils import paths


wd = getcwd()
#print(wd)
#/project/zbwang/TBT/Pytorch/arcface-pytorch-master/data/CASIA-WebFace_Clean&&align


def transform_clean_list(webface_directory, cleaned_list_path):
    """转换webface的干净列表格式
    Args:
        webface_directory: WebFace数据目录
        cleaned_list_path: cleaned_list.txt路径
    Returns:
        cleaned_list: 转换后的数据列表
    """
    with open(cleaned_list_path, encoding='utf-8') as f:
        cleaned_list = f.readlines()
    cleaned_list = [p.replace('\\', '/') for p in cleaned_list]
    cleaned_list = [osp.join(webface_directory, p) for p in cleaned_list]
    return cleaned_list

def save_cleaned_list(list):
    list_file = open('new_cleaned list.txt','w')
    for i in list:
        list_file.write("%s" % (i))

    

def remove_dirty_image(webface_directory, cleaned_list):
    cleaned_list = set([c.split()[0] for c in cleaned_list])
    #imutils.paths的列出图片的功能，列出所有在webface_directory中的图片路径，然后检查这个路径是否在干净列表中，如果不在，我们就删除这张图片
    for p in paths.list_images(webface_directory):
        if p not in cleaned_list:
            print(f"remove {p}")
            os.remove(p)


if __name__ == '__main__':
    data = '/project/zbwang/TBT/Pytorch/arcface-pytorch-master/data/CASIA-WebFace_origin/'
    lst = 'cleaned list.txt'
    cleaned_list = transform_clean_list(data, lst)
    save_cleaned_list(cleaned_list)
    remove_dirty_image(data, cleaned_list)