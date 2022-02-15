#在实例图片上依次画点，以确定MTCNN输出的人脸五点的顺序
#实际的顺序是：左眼，右眼，鼻子，左嘴角，右嘴角。和常规顺序是一致的
import numpy as np
import cv2


origin =    [(0.31556875000000000, 0.4615741071428571),
            (0.68262291666666670, 0.4615741071428571),
            (0.50026249999999990, 0.6405053571428571),
            (0.34947187500000004, 0.8246919642857142),
            (0.65343645833333330, 0.8246919642857142)]

imgSize = [112, 96];

coord5point = []

for x in origin:
    point = []
    point.append(x[0]*96)
    point.append(x[1]*112)
    coord5point.append(point)
    #print(point)

print(coord5point)


'''
cv2.circle(img, points[0], 1, (220,20,60), 4)# 紫色 左眼

#cv2.circle(img, points[1], 1, (128,0,128), 4)# 猩红 右眼

#cv2.circle(img, points[2], 1, (0,0,255), 4)# 橙色 鼻子

#cv2.circle(img, points[3], 1, (46,139,87), 4)# 绿色 左嘴

#cv2.circle(img, points[4], 1, (255,165,0), 4)# 蓝色 右嘴

cv2.circle(img, points[5], 1, (255,165,0), 4)

cv2.namedWindow("image")
cv2.imshow('image', img)
cv2.waitKey (10000) # 显示 10000 ms 即 10s 后消失
cv2.destroyAllWindows()
'''