import os
from os import getcwd

wd = getcwd()
#print(wd)


datasets_path = "64_CASIA-FaceV5/"
types_name = os.listdir(datasets_path)
#print(types_name)
#types_name = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010']


#list_file = open('cls_train.txt','w')
for cls_id,types_name in enumerate(types_name):
    print(cls_id)
    print('\n')
    print(types_name)
    print('\n')
    photos_path = os.path.join(datasets_path,types_name)
    #print(photos_path)
    #print('\n')
    photos_name = os.listdir(photos_path)
    #print(photos_name)
    #print('\n')
    break
    for photo_name in photos_name:
        #print(photo_name)
        #print('\n')
        list_file.write("%s/%s" % (photos_path,photo_name)+" "+str(cls_id))
        list_file.write('\n')
#list_file.close()
