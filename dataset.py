# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
import torch
import h5py
from PIL import Image
from os import listdir
from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, ToPILImage
from io import BytesIO
import numpy as np
import dlib
from utils import get_mask
import time
import os

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.target = hf.get("label")

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        return self.data.shape[0]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', 'jpeg'])

def load_img(filepath):
    # img = Image.open(filepath).convert('YCbCr')
    # y,_,_ = img.split()
    img = Image.open(filepath).convert('RGB')
    return img

def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize((crop_size[0] // upscale_factor, crop_size[1] // upscale_factor)),   #https://stackoverflow.com/questions/48446898/unknown-resampling-filter-error-when-trying-to-create-my-own-dataset-with-pytorc
        # ToTensor(),   # 不使用ToTensor，transform后仍然返回PIL.Image对象
    ])      # 实例化Compose，将变换函数列表传入构造函数

def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        # ToTensor(),
    ])

def compress(im, quality=20):
    # https://stackoverflow.com/questions/31409506/python-convert-from-png-to-jpg-without-saving-file-to-disk-using-pil
    f = BytesIO()
    im.save(f, 'JPEG', quality=quality, optimize=True, progressive=True)
    return Image.open(f)

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('/media/lab/data/hanchi/dlib/shape_predictor_68_face_landmarks.dat')

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,  quality=20, input_transform=None, target_transform=None, face_detector=face_detector, landmark_predictor=landmark_predictor,):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir,x) for x in listdir(image_dir) if is_image_file(x)]
        self.face_detector = face_detector
        self.landmark_predictor = landmark_predictor
        self.quality = quality

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        # print(self.image_filenames[index])
        # basename = os.path.basename(self.image_filenames[index])
        # mask_name = os.path.splitext(basename)[0]
        target = input.copy()

        input = compress(input, self.quality)

        if self.input_transform:
            input = self.input_transform(input)     # 经过transforms.ToTensor后，图像scale从[0,255]转为[0,1]
            # input = compress(input, self.quality)
            lr_mask = get_mask(input, self.face_detector, self.landmark_predictor)
            lr_mask = lr_mask[np.newaxis, :]
            input = ToTensor()(input)
        if self.target_transform:
            target = self.target_transform(target)
            # start_time = time.time()
            hr_mask = get_mask(target, self.face_detector, self.landmark_predictor)
            # elapsed_time = time.time() - start_time
            # print ('time {}s'.format(elapsed_time))
            hr_mask = hr_mask[np.newaxis, :]
            target = ToTensor()(target)
            # print 'type ', type(target)
            # cv2.imshow('cropped',np.array(target)[0,:,:])
            # cv2.waitKey(0)
            return input, target, lr_mask, hr_mask

    def __len__(self):
        return len(self.image_filenames)

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def get_training_set(train_dir, crop_size, upscale_factor, quality):
    # crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
    return DatasetFromFolder(train_dir, quality,
                             input_transform = input_transform(crop_size, upscale_factor),
                             target_transform = target_transform(crop_size))

def get_test_set(test_dir, crop_size, upscale_factor, quality):
    # crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
    return DatasetFromFolder(test_dir, quality,
                             input_transform = input_transform(crop_size, upscale_factor),
                             target_transform = target_transform(crop_size))

if __name__ == "__main__":
    image_dir = 'data/img_align_celeba'
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('/media/lab/data/hanchi/dlib/shape_predictor_68_face_landmarks.dat')
    dataset = DatasetFromFolder(image_dir, 20, input_transform((128,128),2), target_transform(128))
    input, target, lr_mask, hr_mask = dataset.__getitem__(0)

    # print(get_test_set)