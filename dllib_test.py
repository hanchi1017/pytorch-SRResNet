# coding=utf-8
import cv2
import dlib
import numpy as np
# https://www.cnblogs.com/as3asddd/p/7257820.html
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('/media/lab/data/hanchi/dlib/shape_predictor_68_face_landmarks.dat')
img = cv2.imread('/media/lab/data/hanchi/PycharmProjects/pytorch-SRResNet/data/face2.jpg')
faces = detector(img,1)


if (len(faces) > 0):
    for k,d in enumerate(faces):
        nose = []
        left_eye = []
        right_eye = []
        mouth = []

        cv2.rectangle(img, (d.left(),d.top()), (d.right(),d.bottom()), (255,255,255))
        shape = landmark_predictor(img, d)

        for i in range(68):
            cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0,255,255), -1, 8)
            cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        for i in range(27,36):
            nose.append([shape.part(i).x,shape.part(i).y])

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

        print np.array(nose).shape
        nose = np.array(nose)
        mouth = np.array(mouth)
        left_eye = np.array(left_eye)
        right_eye = np.array(right_eye)
        cv2.rectangle(img, (right_eye[:,0].max(),nose[:,1].min()), (left_eye[:,0].min(),nose[:,1].max()), (0,255,0))    # 确定鼻子的高频区域
        cv2.rectangle(img, (mouth[:,0].min(),mouth[:,1].min()), (mouth[:,0].max(),mouth[:,1].max()), (0,255,0))
        cv2.rectangle(img, (left_eye[:,0].min(),left_eye[:,1].min()), (left_eye[:,0].max(),left_eye[:,1].max()), (0,255,0))
        cv2.rectangle(img, (right_eye[:,0].min(),right_eye[:,1].min()), (right_eye[:,0].max(),right_eye[:,1].max()), (0,255,0))

        # ellipse_nose = cv2.fitEllipse(np.array(nose).astype(np.int32))
        # ellipse_left = cv2.fitEllipse(np.array(left_eye))
        # ellipse_right = cv2.fitEllipse(np.array(right_eye))
        #
        # cv2.ellipse(img, ellipse_nose, (0,255,0), 1)
        # cv2.ellipse(img, ellipse_left, (0,255,0), 1)
        # cv2.ellipse(img, ellipse_right, (0,255,0), 1)

cv2.imshow('Frame',img)
cv2.waitKey(0)