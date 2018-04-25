import numpy as np
import math
from os.path import isdir, isfile, join
from os import remove, listdir
import cv2
import dlib
from PIL import Image

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def delete_all(path_list):
    for path in path_list:
        if isfile(path):
            remove(path)
        elif isdir(path):
            for x in listdir(path):
                remove(join(path, x))

def get_mask(img):
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('/media/lab/data/hanchi/dlib/shape_predictor_68_face_landmarks.dat')

    img = np.array(img)
    mask = np.zeros(img.shape[0:2], dtype=np.float32)
    # print 'mask shape: ',mask.shape
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    img = cv2.merge([b, g, r])
    faces = detector(img, 1)

    if(len(faces)>0):
        for k,d in enumerate(faces):
            nose = []
            left_eye = []
            right_eye = []
            mouth = []
            shape = landmark_predictor(img,d)
            for i in range(27,36):
                nose.append([shape.part(i).x, shape.part(i).y])
            for i in range(48,60):
                mouth.append([shape.part(i).x, shape.part(i).y])

            for i in range(17,22):
                right_eye.append([shape.part(i).x, shape.part(i).y])
            for i in range(36,42):
                right_eye.append([shape.part(i).x, shape.part(i).y])

            for i in range(22,27):
                left_eye.append([shape.part(i).x, shape.part(i).y])
            for i in range(42,48):
                left_eye.append([shape.part(i).x, shape.part(i).y])

            nose = np.array(nose)
            mouth = np.array(mouth)
            left_eye = np.array(left_eye)
            right_eye = np.array(right_eye)

            mask[nose[:,1].min():nose[:,1].max(), right_eye[:,0].max():left_eye[:,0].min()] = 1
            mask[mouth[:,1].min():mouth[:,1].max(), mouth[:,0].min():mouth[:,0].max()] = 1
            mask[left_eye[:,1].min():left_eye[:,1].max(), left_eye[:,0].min():left_eye[:,0].max()] = 1
            mask[right_eye[:,1].min():right_eye[:,1].max(), right_eye[:,0].min():right_eye[:,0].max()] =1
    # ret,thresh = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY_INV)
    # print type(thresh)
    # cv2.imshow('jj', thresh)
    # cv2.waitKey(0)
    return mask


if __name__ == "__main__":
    filepath = '/media/lab/data/hanchi/PycharmProjects/pytorch-SRResNet/data/face2.jpg'
    img = Image.open(filepath)
    print get_mask(img).shape