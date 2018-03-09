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

parser = argparse.ArgumentParser(description="PyTorch SRResNet Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_415.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"]

# im_gt = sio.loadmat("Set5/" + opt.image + ".mat")['im_gt']    # 类型为float，范围[0,255]
# im_b = sio.loadmat("Set5/" + opt.image + ".mat")['im_b']
# im_l = sio.loadmat("Set5/" + opt.image + ".mat")['im_l']

test_dir = 'data/testset'   # 包含原始gt的image
# test_dir = 'data/other_method/img_gt'
hr_size = 128
testset = get_test_set(test_dir, hr_size, upscale_factor=4, quality=40)
im_l, im_gt = testset.__getitem__(0)

im_l = im_l.numpy().astype(np.float32)  # 类型float，范围[0,1]

im_b = (im_l*255.).astype(np.uint8)
im_b = smi.imresize(im_b, (hr_size, hr_size, 3), interp='bicubic')

im_gt = im_gt.numpy().transpose(1,2,0)      # 为了显示需要做一次转置
im_gt = (im_gt.astype(float)*255.).astype(np.uint8)

im_input = im_l.reshape(1,im_l.shape[0],im_l.shape[1],im_l.shape[2])
im_input = Variable(torch.from_numpy(im_input).float())

if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
else:
    model = model.cpu()
    
start_time = time.time()
out = model(im_input)
elapsed_time = time.time() - start_time

out = out.cpu()

im_h = out.data[0].numpy().astype(np.float32)

im_h = im_h*255.
im_h[im_h<0] = 0
im_h[im_h>255.] = 255.            
im_h = im_h.transpose(1,2,0).astype(np.uint8)

# 转换为灰度图，计算PSNR（即只计算Y分量）
# im_gt_grey = Image.fromarray(im_gt)
im_gt_grey = np.array(smi.toimage(im_gt).convert('L'))  # 函数toimage 将numpy array转为PIL.Image.Image, 或者使用Image.fromarray
im_b_grey = np.array(smi.toimage(im_b).convert('L'))
im_h_grey = np.array(smi.toimage(im_h).convert('L'))

psnr_bicubic = PSNR(im_b_grey, im_gt_grey)
psnr_output = PSNR(im_h_grey, im_gt_grey)

print("Scale=",opt.scale)
print("It takes {}s for processing".format(elapsed_time))
print("bicubic psnr : {} ; net output psnr : {}".format(psnr_bicubic, psnr_output))

fig = plt.figure()
ax = plt.subplot("131")
ax.imshow(im_gt)
ax.set_title("GT")

ax = plt.subplot("132")
ax.imshow(im_b)
ax.set_title("Input(Bicubic)")

ax = plt.subplot("133")
ax.imshow(im_h)
ax.set_title("Output(Net)")
plt.show()
