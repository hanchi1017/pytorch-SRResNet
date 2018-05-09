# coding=utf-8
from __future__ import print_function
import numpy as np
import math
import os
from os.path import isdir, isfile, join, exists
from os import remove, listdir, mkdir
import cv2
import dlib
from PIL import Image
import time

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
    for path in path_list[:-1]:
        if not exists(path):
            mkdir(path)
    for path in path_list:
        if isfile(path):
            remove(path)
        elif isdir(path):
            for x in listdir(path):
                remove(join(path, x))

def get_mask(img, face_detector, landmark_predictor):

    img = np.array(img)
    mask = np.zeros(img.shape[0:2], dtype=np.float32)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    img = cv2.merge([b, g, r])
    faces = face_detector(img, 1)

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
    ret,thresh = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY_INV)
    # print type(thresh)
    # cv2.imshow('jj', thresh)
    # cv2.waitKey(0)
    # return mask

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])

def save_mask_batch(img_dir, mask_dir):
    for x in listdir(img_dir):
        start_time = time.time()
        if is_image_file(x):
            img = Image.open(join(img_dir, x))
            mask = get_mask(img, face_detector, landmark_predictor)
            np.save(join(mask_dir, os.path.splitext(x)[0]), mask)
            elapsed_time = time.time() - start_time
            # print('{} is saved, spent {}s'.format(x, elapsed_time))

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('/media/lab/data/hanchi/dlib/shape_predictor_68_face_landmarks.dat')

if __name__ == "__main__":
    # filepath = 'data/testset_glass'
    # img = Image.open(filepath)
    # mask = get_mask(img, face_detector, landmark_predictor)
    # print(mask.shape)

    img = cv2.imread('data/testset_glass/002058.jpg')
    faces = face_detector(img, 1)
    if (len(faces) > 0):
        for k, d in enumerate(faces):
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255))
            shape = landmark_predictor(img, d)
            for i in range(68):
                cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0), -1, 8)
                # cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 2555, 255))
    cv2.imshow('Frame', img)
    cv2.waitKey(0)

    # img_dir = '/media/lab/data/hanchi/PycharmProjects/pytorch-SRResNet/data/img_align_celeba'
    # # img_dir = '/media/lab/data/hanchi/PycharmProjects/pytorch-SRResNet/data/other_method/img_gt'
    # mask_dir = '/media/lab/data/hanchi/PycharmProjects/pytorch-SRResNet/data/img_align_celeba_mask'
    # save_mask_batch(img_dir, mask_dir)