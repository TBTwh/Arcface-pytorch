import os
import cv2



pic_path = 'C:\\Users\\TBT\\Desktop\\64_CASIA-FaceV5'
# 三种不同的路径表达方式
root = r'C:\Users\TBT\Desktop\results'     # 绝对路径的写法
#root = 'C:\\Users\\TBT\\Desktop\\results'   # 显式声明字符串不用转义（加r）
#root = 'C:/Users/TBT/Desktop/results'# 使用Linux的路径/

#img = cv2.imread("C:\\Users\\TBT\\Desktop\\000/000_1.bmp")
pic_1_catalogue_list = os.listdir(pic_path)  #['000', '001',···'499']
print("pic_1_catalogue_list = {}\n".format(pic_1_catalogue_list))
for pic_1_catalogue in pic_1_catalogue_list:
    #print('pic_1_catalogue = {}\n'.format(pic_1_catalogue))
    target_path_1_catalogue = os.path.join(root,pic_1_catalogue)
    origin_path_1_catalogue = os.path.join(pic_path,pic_1_catalogue)
    
    
    if(os.path.isdir(target_path_1_catalogue)==0):
        os.mkdir(target_path_1_catalogue)  # 创建单级目录
        #os.makedirs(target_path_1_catalogue) # 创建多级目录(缺失多级目录会连续创建)    

    origin_pic_list = os.listdir(origin_path_1_catalogue)
    for every_pic_name in origin_pic_list:
        #print("every_pic_name = {}".format(every_pic_name))
        path_item = os.path.join(target_path_1_catalogue,every_pic_name)
        #print("path_item = {}".format(path_item))
        
        img_path = os.path.join(origin_path_1_catalogue,every_pic_name)
        print("img_path = {}".format(img_path))
        


'''
pic_name_list = os.listdir(pic_path)
print(pic_name_list)
'''

'''
for dirpath, dirnames, filenames in os.walk(pic_path):
    print(dirpath)
    print(dirnames)
    print(filenames)
    print("\n\n")
'''

#root = 'C:\Users\TBT\Desktop\results'

'''
#path = 'C:\\Users\\TBT\\Desktop\\64_CASIA-FaceV5'
path = 'C:\\Users\\TBT\\Desktop\\64_CASIA-FaceV5\\000/000_1.bmp'
print(os.path.isfile(path))
print(os.path.isdir(path))
'''



