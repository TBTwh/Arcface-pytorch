# mtcnn_test 用于验证mtcnn的正常运行
import tensorflow as tf
from mtcnn import MTCNN
import cv2


'''
归一化五点坐标
[(0.31556875000000000, 0.4615741071428571),
 (0.68262291666666670, 0.4615741071428571),
 (0.50026249999999990, 0.6405053571428571),
 (0.34947187500000004, 0.8246919642857142),
 (0.65343645833333330, 0.8246919642857142)]
'''

print("hello")

#img = cv2.imread('D:\\lab\\arcface-pytorch-master\\data\\64_CASIA-FaceV5_origin\\258\\258_2.bmp')
img = cv2.imread('/home/zbwang/project/TBT/Pytorch/arcface-pytorch-master/data/64_CASIA-FaceV5_origin/000/000_0.bmp')
# img.shape(480, 640, 3)

detector = MTCNN()
face = detector.detect_faces(img)

print(face)
'''
[{'box': [317, 176, 87, 106], 
'confidence': 0.998997151851654, 
'keypoints': {'left_eye': (339, 219), 'right_eye': (377, 215), 'nose': (359, 239), 'mouth_left': (346, 258), 'mouth_right': (380, 254)}}]
'''

print("\n\n")
#默认只有一张人脸

if(len(face)>1):
    print('len(face) = {}'.format(len(face)))
    confidence = 0
    tar = 0
    for i in range(0,len(face)):
        #print('i={}'.format(i))
        tmp_face = face[i]
        #print('tmp_face["confidence"]={}'.format(tmp_face["confidence"]))
        #print('confidence={}'.format(confidence))
        if(tmp_face["confidence"]>confidence):
            confidence = tmp_face["confidence"]
            tar = i
    print('tar={}'.format(tar))  
    face = face[tar]    
else:
    face = face[0]
'''

face = face[1]
'''
print(face)
print(face["confidence"])

#画框
box = face["box"]

I = cv2.rectangle(img, (box[0],box[1]),(box[0]+box[2], box[1]+box[3]), (255, 0, 0), 2)
#cv2.rectangle(image, start_point, end_point, color, thickness)

#画关键点
left_eye = face["keypoints"]["left_eye"]
right_eye = face["keypoints"]["right_eye"]
nose = face["keypoints"]["nose"]
mouth_left = face["keypoints"]["mouth_left"]
mouth_right = face["keypoints"]["mouth_right"]

points_list = [(left_eye[0], left_eye[1]),
               (right_eye[0], right_eye[1]),
               (nose[0], nose[1]),
               (mouth_left[0], mouth_left[1]),
               (mouth_right[0], mouth_right[1])]
for point in points_list:
	cv2.circle(I, point, 1, (255, 0, 0), 4)
    #cv2.circle(image, center_coordinates, radius, color, thickness)
#保存
cv2.imwrite('result.bmp',I,[int(cv2.IMWRITE_JPEG_QUALITY),70])


