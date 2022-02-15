'''
----TBT 2022/1/30----
实现了对原始图片的人脸提取和对齐
#ratio 功能是对点映射时候的文件做外扩，对最后人脸图片的大小无意义
把原图中人脸区域外扩100%（这样做的目的是保证对齐后图片中没有黑色区域，当然这个外扩的比例是看对齐效果自己可以调节的，我这里设置的100%）
'''
import os
import cv2
import numpy
import tensorflow as tf
from mtcnn import MTCNN





def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),numpy.matrix([0., 0., 1.])])
 
def warp_im(img_im, orgi_landmarks,tar_landmarks):
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst



def coord5point_calculation(imgSize):
    origin =    [(0.31556875000000000, 0.4615741071428571),
                (0.68262291666666670, 0.4615741071428571),
                (0.50026249999999990, 0.6405053571428571),
                (0.34947187500000004, 0.8246919642857142),
                (0.65343645833333330, 0.8246919642857142)]
    coord5point = []
    for x in origin:
        point = []
        point.append(x[0]*imgSize[1])
        point.append(x[1]*imgSize[0])
        coord5point.append(point)  
    return coord5point


def get_align_img(img,coord5point,ratio,path):
    shape = img.shape
    height = shape[0]
    width = shape[1]

    detector = MTCNN()
    face = detector.detect_faces(img)
    for i in range(len(face)):
        i_face = face[i]
        box = i_face["box"]

        #得到人脸五点x,y坐标
        left_eye = i_face["keypoints"]["left_eye"]
        right_eye = i_face["keypoints"]["right_eye"]
        nose = i_face["keypoints"]["nose"]
        mouth_left = i_face["keypoints"]["mouth_left"]
        mouth_right = i_face["keypoints"]["mouth_right"]
        x = [left_eye[0],right_eye[0],nose[0],mouth_left[0],mouth_right[0]]
        y = [left_eye[1],right_eye[1],nose[1],mouth_left[1],mouth_right[1]]

        new_coor = extend_coordinate(box,x,y,ratio,width,height)
        bounding_start,bounding_end,face_landmarks = new_coor[0],new_coor[1],new_coor[2]

        crop_face = img[bounding_start[1]: bounding_end[1], bounding_start[0]: bounding_end[0]] # 按照新的bounding裁剪原图
        dst = warp_im(crop_face,face_landmarks,coord5point) # 按照imgSize旋转对齐
        crop_im = dst[0:imgSize[0],0:imgSize[1]] #裁剪到imgSize大小

        my_save(path,crop_im)
        
def my_save(path,img):
    cv2.imwrite(path,img,[int(cv2.IMWRITE_JPEG_QUALITY),70])

# 传入bounding_coor:box = face["box"]
# 传入bounding_coor:points = [x,y]
# x = [left_eye,right_eye,nose,left_mouse,right_mouse]
# y = [left_eye,right_eye,nose,left_mouse,right_mouse]
# ratio[2]:原来五点人脸img的x,y放缩比例 ratio[0]:x    ratio[1]:y
def extend_coordinate(box,x,y,ratio,width,height):
    box_start_point = []
    box_start_point.append(box[0])
    box_start_point.append(box[1])
    box_end_point = []
    box_end_point.append(box[0]+box[2])
    box_end_point.append(box[1]+box[3])
    
    x1, y1, x2, y2 =    int(min(box_start_point[0], min(x))), \
                        int(min(box_start_point[1], min(y))), \
                        int(max(box_end_point[0], max(x))), \
                        int(max(box_end_point[1], max(y)))
    #按缩放比例扩大box
    new_x1 = max(int(x1 - ratio[0]*(x2 - x1)),0)
    new_x2 = min(int(x2 + ratio[0]*(x2 - x1)),width-1)
    new_y1 = max(int(y1 - ratio[1]*(y2 - y1)),0)
    new_y2 = min(int(y2 + ratio[1]*(y2 - y1)),height-1)

    # 得到原始图中关键点坐标
    left_eye_x = x[0]
    right_eye_x = x[1]
    nose_x = x[2]
    left_mouth_x = x[3]
    right_mouth_x = x[4]
    left_eye_y = y[0]
    right_eye_y = y[1]
    nose_y = y[2]
    left_mouth_y = y[3]
    right_mouth_y = y[4]

    # 得到外扩ratio比例后图中关键点坐标(之后会按照start点作为开始裁剪原本的img)
    new_left_eye_x = left_eye_x - new_x1
    new_right_eye_x = right_eye_x - new_x1
    new_nose_x = nose_x - new_x1
    new_left_mouth_x = left_mouth_x - new_x1
    new_right_mouth_x = right_mouth_x - new_x1
    new_left_eye_y = left_eye_y - new_y1
    new_right_eye_y = right_eye_y - new_y1
    new_nose_y = nose_y - new_y1
    new_left_mouth_y = left_mouth_y - new_y1
    new_right_mouth_y = right_mouth_y - new_y1

    # 在按照ratio扩大后人脸图中关键点坐标
    face_landmarks =    [[new_left_eye_x,new_left_eye_y], 
                        [new_right_eye_x,new_right_eye_y],
                        [new_nose_x,new_nose_y],
                        [new_left_mouth_x,new_left_mouth_y],
                        [new_right_mouth_x,new_right_mouth_y]]
    
    bounding_start = [new_x1,new_y1]
    bounding_end = [new_x2,new_y2]
    ans = [bounding_start,bounding_end,face_landmarks]
    return ans    

if __name__ == '__main__':
    imgSize = [128, 128]    
    coord5point = coord5point_calculation(imgSize)
    ratio = [1,1]#[100%,100%]

    # 此部分仅仅适用于原图片集目录下含一级子目录的情况
    # pic_path 为原图片集总目录
    # root 为人脸对齐后的图片集总目录
    pic_path = '/project/zbwang/TBT/Pytorch/arcface-pytorch-master/data/CASIA-WebFace_origin/'
    root = '/project/zbwang/TBT/Pytorch/arcface-pytorch-master/data/CASIA-WebFace/'
    #pic_path = 'D:\\lab\\arcface-pytorch-master\\data\\64_CASIA-FaceV5_origin'
    #root = 'D:\\lab\\arcface-pytorch-master\\data\\64_CASIA-FaceV5'     # 绝对路径的写法
    
    pic_1_catalogue_list = os.listdir(pic_path)  #['000', '001',···'499']
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
            img_path = os.path.join(origin_path_1_catalogue,every_pic_name)
            img = cv2.imread(img_path)
            path_item = os.path.join(target_path_1_catalogue,every_pic_name)            
            get_align_img(img,coord5point,ratio,path_item)

    '''
    #save_test
    img = cv2.imread("C:\\Users\\TBT\\Desktop\\64_CASIA-FaceV5\\000\\000_1.bmp")
    path = 'C:\\Users\\TBT\\Desktop\\results\\000_1.bmp'
    get_align_img(img,coord5point,ratio,path)
    '''
    


    
    
    cv2.waitKey()
    pass
