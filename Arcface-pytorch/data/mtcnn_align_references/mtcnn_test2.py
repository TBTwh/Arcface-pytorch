# mtcnn_test2 用于验证利用五点坐标对齐人脸的程序
#coding=utf-8
import os,cv2,numpy
import logging
logging.basicConfig(
	level=logging.DEBUG,
	format='%(asctime)s %(levelname)s: %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

#imgSize = [112, 96];
imgSize = [224, 192];
coord5point = [[30.2946, 51.6963],
               [65.5318, 51.6963],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.3655]]


# TBT 2022/1/26
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
    #print(point)        



face_landmarks = [[339, 219],
                  [377, 215],
                  [359, 239],
                  [346, 258],
                  [380, 254]]


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

def main():
    pic_path = r'C:\Users\TBT\Desktop\000\000_1.bmp'
    img_im = cv2.imread(pic_path)
    cv2.imshow('affine_img_im', img_im)
    dst = warp_im(img_im, face_landmarks, coord5point)
    cv2.imshow('affine', dst)
    crop_im = dst[0:imgSize[0], 0:imgSize[1]]
    cv2.imshow('affine_crop_im', crop_im)

if __name__=='__main__':
    main()
    cv2.waitKey()
    pass

