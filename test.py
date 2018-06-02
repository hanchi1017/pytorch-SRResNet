# coding=utf-8
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import scipy.misc as smi
import matplotlib.pyplot as plt
from dataset import get_test_set
from PIL import Image
from os.path import join, isdir, isfile, exists
from os import listdir, remove
from utils import PSNR, delete_all
import cv2,dlib
from torch.utils.data import DataLoader
import os

parser = argparse.ArgumentParser(description="PyTorch SRResNet Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_415.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"]
epoch = torch.load(opt.model)["epoch"]
if cuda:
    model = model.cuda()
else:
    model = model.cpu()

test_dir = 'data/gt'   # 包含原始gt的image
hr_size = (120, 100)
# test_dir = 'data/other_method/img_gt'
# hr_size = (320,240)     # (height, width)
root_dir = 'result/result_1'
if not exists(root_dir):
    os.mkdir(root_dir)
else:
    raise Exception("result already exits")

gt_dir = join(root_dir,'img_gt')
lr_dir = join(root_dir,'img_lr')
hr_dir = join(root_dir,'img_hr_{}'.format(epoch))
bi_dir = join(root_dir,'img_bi')
gt_dir_grey = join(root_dir,'img_gt_grey')
lr_dir_grey = join(root_dir,'img_lr_grey')
hr_dir_grey = join(root_dir,'img_hr_grey')
bi_dir_grey = join(root_dir,'img_bi_grey')
list_txt = join(hr_dir,'result_epoch_{}.txt'.format(epoch))

path_list = [gt_dir, lr_dir, hr_dir, bi_dir, gt_dir_grey, lr_dir_grey, hr_dir_grey, bi_dir_grey, list_txt]
delete_all(path_list)

testset = get_test_set(test_dir, hr_size, upscale_factor=4, quality=40)     # 得到测试集图像
test_data_loader = DataLoader(dataset=testset, batch_size=32)


for i in range(0, testset.__len__()):
# for iteration, batch in enumerate(test_data_loader, 1):
    im_l, im_gt, prefix,  _, _ = testset.__getitem__(i)

    im_l = im_l.numpy().astype(np.float32)  # 类型float，范围[0,1]
    im_lr_norm = (im_l*255.).astype(np.uint8).transpose(1,2,0)
    im_b = smi.imresize(im_lr_norm, (hr_size[0], hr_size[1], 3), interp='bicubic')

    im_gt = im_gt.numpy().transpose(1,2,0)      # 为了保存、显示需要做一次转置
    im_gt = (im_gt.astype(float)*255.).astype(np.uint8)

    im_input = im_l.reshape(1,im_l.shape[0],im_l.shape[1],im_l.shape[2])
    im_input = Variable(torch.from_numpy(im_input).float())
    if cuda:
        im_input = im_input.cuda()

    start_time = time.time()
    out = model(im_input)
    elapsed_time = time.time() - start_time

    out = out.cpu()

    im_h = out.data[0].numpy().astype(np.float32)
    im_h = im_h*255.
    im_h[im_h<0] = 0
    im_h[im_h>255.] = 255.
    im_h = im_h.transpose(1,2,0).astype(np.uint8)

    # 保存图像到各自目录
    smi.imsave(join(gt_dir, '{}.png'.format(prefix)), im_gt)
    smi.imsave(join(hr_dir, '{}.png'.format(prefix)), im_h)
    smi.imsave(join(bi_dir, '{}.png'.format(prefix)), im_b)
    smi.imsave(join(lr_dir, '{}.png'.format(prefix)), im_lr_norm)  # LR image id saved here

    # 转换为灰度图，计算PSNR（即只计算Y分量）
    # im_gt_grey = Image.fromarray(im_gt)
    im_gt_grey = np.array(smi.toimage(im_gt).convert('L'))  # 函数toimage 将numpy array转为PIL.Image.Image, 或者使用Image.fromarray
    im_bi_grey = np.array(smi.toimage(im_b).convert('L'))
    im_hr_grey = np.array(smi.toimage(im_h).convert('L'))
    im_lr_grey = np.array(smi.toimage(im_lr_norm).convert('L'))
    smi.imsave(join(gt_dir_grey, '{}.png'.format(prefix)), im_gt_grey)
    smi.imsave(join(bi_dir_grey, '{}.png'.format(prefix)), im_bi_grey)
    smi.imsave(join(hr_dir_grey, '{}.png'.format(prefix)), im_hr_grey)
    smi.imsave(join(lr_dir_grey, '{}.png'.format(prefix)), im_lr_grey)
    print('images are saved')

    psnr_bicubic = PSNR(im_bi_grey, im_gt_grey)
    psnr_output = PSNR(im_hr_grey, im_gt_grey)

    print("{}th image, Scale={}".format(i, opt.scale))
    print("It takes {}s for processing".format(elapsed_time))
    print("bicubic psnr : {} ; ours psnr : {}\n".format(psnr_bicubic, psnr_output))

    with open(list_txt,'a') as f:
        f.write('{} {}.png bicubic psnr: {} ; ours psnr: {}\n'.format(i,i,psnr_bicubic,psnr_output))
    #
    # fig = plt.figure()
    # ax = plt.subplot("131")
    # ax.imshow(im_gt)
    # ax.set_title("GT")
    #
    # ax = plt.subplot("132")
    # ax.imshow(im_b)
    # ax.set_title("Output(Bicubic)\nPSNR={}".format(psnr_bicubic))
    #
    # ax = plt.subplot("133")
    # ax.imshow(im_h)
    # ax.set_title("Output(Net)\nPSNR={}".format(psnr_output))

    # ax = plt.subplot("144")
    # ret, thresh = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY_INV)
    # ax.imshow(thresh)
    # ax.set_title("high frequency")
    # plt.show()
