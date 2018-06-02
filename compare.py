# coding=utf-8
import scipy.misc as smi
from skimage.measure import compare_ssim as ssim
import numpy as np
import math
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import argparse
from utils import PSNR, delete_all
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="color", type=str, help="color or grey?")

opt = parser.parse_args()
type = opt.type

gt_dir = ''
lr_dir = ''
hr_dir = ''
bi_dir = ''
SFH_dir = ''    # 存储Structured Face Hallucination中方法的结果
SSR_dir = ''
SRCNN_dir = ''
V1_dir = ''
V2_dir = ''
bi_avg_psnr = 0
SSR_avg_psnr = 0
SRCNN_avg_psnr = 0
V1_avg_psnr = 0
V2_avg_psnr = 0

bi_avg_ssim = 0
SSR_avg_ssim = 0
SRCNN_avg_ssim = 0
V1_avg_ssim = 0
V2_avg_ssim = 0

root_dir = 'result/compare_dir/compare_1'
if not os.path.exists(root_dir):
    os.mkdir(root_dir)
perImgResult = open(join(root_dir,'perImgResult.csv'), 'a')
summaryResult = open(join(root_dir,'summaryResult.csv'), 'a')
fieldnames1 = ['img_name','bicubic','SSR','SRCNN','ours_v1','ours_v2']
fieldnames2 = ['Q=40','bicubic','SSR','SRCNN','ours_v1','ours_v2']
perImgWriter = csv.DictWriter(perImgResult, fieldnames=fieldnames1)
summaryWriter = csv.DictWriter(summaryResult, fieldnames=fieldnames2)
perImgWriter.writeheader()
summaryWriter.writeheader()

if type == "color":
    gt_dir = join(root_dir,'gt')
    lr_dir = join(root_dir,'lr')
    bi_dir = join(root_dir,'bi')
    SSR_dir = join(root_dir, 'SSR')
    SRCNN_dir = join(root_dir, 'SRCNN')
    V1_dir = join(root_dir, 'V1')
    V2_dir = join(root_dir, 'V2')
    #SFH_dir = '/media/lab/data/hanchi/Structured_Face_Hallucination-v1.4/Code/Ours2/Result/Test66_ReOrganizeCVPR13' join(root_dir,'img_gt')
    #compare_dir = 'result/img_compare'  join(root_dir,'img_gt')
# elif type == "grey":
#     gt_dir = 'result/img_gt_grey'
#     lr_dir = 'result/img_lr_grey'
#     hr_dir = 'result/img_hr_grey'
#     bi_dir = 'result/img_bi_grey'
#     SFH_dir = '/media/lab/data/hanchi/Structured_Face_Hallucination-v1.4/Code/Ours2/Result/Test66_ReOrganizeCVPR13'
#     SSR_dir = 'result/img_SSR'
#     compare_dir = 'result/img_compare_grey'

#delete_all([compare_dir])   # 首先将对比结果的目录清空
filenames = listdir(gt_dir)
print(filenames)
for filename in filenames:
    print(filename)
    im_gt = smi.imread(join(gt_dir, filename))
    im_bi = smi.imread(join(bi_dir, filename))
    im_SSR = smi.imread(join(SSR_dir, filename))
    im_SRCNN = smi.imread(join(SRCNN_dir, filename))
    im_V1 = smi.imread(join(V1_dir, filename))
    im_V2 = smi.imread(join(V2_dir, filename))

    im_gt_grey = np.array(smi.toimage(im_gt).convert('L'))
    im_bi_grey = np.array(smi.toimage(im_bi).convert('L'))
    im_SSR_grey = np.array(smi.toimage(im_SSR).convert('L'))
    im_SRCNN_grey = np.array(smi.toimage(im_SRCNN).convert('L'))
    im_V1_grey = np.array(smi.toimage(im_V1).convert('L'))
    im_V2_grey = np.array(smi.toimage(im_V2).convert('L'))

    psnr_bi = PSNR(im_bi_grey, im_gt_grey)
    psnr_SSR = PSNR(im_SSR_grey, im_gt_grey)
    psnr_SRCNN = PSNR(im_SRCNN_grey, im_gt_grey)
    psnr_v1 = PSNR(im_V1_grey, im_gt_grey)
    psnr_v2 = PSNR(im_V2_grey, im_gt_grey)

    ssim_bi = ssim(im_bi_grey, im_gt_grey)
    ssim_SSR = ssim(im_SSR_grey, im_gt_grey)
    ssim_SRCNN = ssim(im_SRCNN_grey, im_gt_grey)
    ssim_v1 = ssim(im_V1_grey, im_gt_grey)
    ssim_v2 = ssim(im_V2_grey, im_gt_grey)

    perImgWriter.writerow({'img_name':filename, 'bicubic':'{:.2f}/{:.3f}'.format(psnr_bi,ssim_bi), 'SSR':'{:.2f}/{:.3f}'.format(psnr_SSR,ssim_SSR),
                           'SRCNN':'{:.2f}/{:.3f}'.format(psnr_SRCNN,ssim_SRCNN), 'ours_v1':'{:.2f}/{:.3f}'.format(psnr_v1,ssim_v1), 'ours_v2':'{:.2f}/{:.3f}'.format(psnr_v2,ssim_v2)})

    bi_avg_psnr += psnr_bi
    SSR_avg_psnr += psnr_SSR
    SRCNN_avg_psnr += psnr_SRCNN
    V1_avg_psnr += psnr_v1
    V2_avg_psnr += psnr_v2

    # http://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
    bi_avg_ssim += ssim_bi
    SSR_avg_ssim += ssim_SSR
    SRCNN_avg_ssim += ssim_SRCNN
    V1_avg_ssim += ssim_v1
    V2_avg_ssim += ssim_v2

summaryWriter.writerow({'Q=40':'PSNR', 'bicubic':'{:.2f}'.format(bi_avg_psnr/len(filenames)), 'SSR':'{:.2f}'.format(SSR_avg_psnr/len(filenames)), 'SRCNN':'{:.2f}'.format(SRCNN_avg_psnr/len(filenames)),
                        'ours_v1':'{:.2f}'.format(V1_avg_psnr/len(filenames)), 'ours_v2':'{:.2f}'.format(V2_avg_psnr/len(filenames))})
summaryWriter.writerow({'Q=40':'SSIM', 'bicubic':'{:.3f}'.format(bi_avg_ssim/len(filenames)), 'SSR':'{:.3f}'.format(SSR_avg_ssim/len(filenames)), 'SRCNN':'{:.3f}'.format(SRCNN_avg_ssim/len(filenames)),
                        'ours_v1':'{:.3f}'.format(V1_avg_ssim/len(filenames)), 'ours_v2':'{:.3f}'.format(V2_avg_ssim/len(filenames))})
perImgResult.close()
summaryResult.close()
print('对比结果保存至{}'.format(root_dir))
    # if type == 'color':
    #     fig = plt.figure()
    #     ax = plt.subplot("221")
    #     ax.imshow(im_gt)
    #     ax.set_title("GT")
    #
    #     ax = plt.subplot("222")
    #     ax.imshow(im_bi)
    #     ax.set_title("Output(Bicubic)\nPSNR={}".format(psnr_bi))
    #
    #     ax = plt.subplot("223")
    #     ax.imshow(im_hr)
    #     ax.set_title("Output(Ours)\nPSNR={}".format(psnr_ours))
    #
    #     ax = plt.subplot("224")
    #     ax.imshow(im_another)
    #     ax.set_title("Output(SFH)\nPSNR={}".format(psnr_another))
    #     plt.tight_layout()
    #     plt.show()
    #     plt.savefig('{}/{}.png'.format(compare_dir, filename.split('.')[0]))
    # elif type == 'grey':
    #     fig = plt.figure()
    #     ax = plt.subplot("221")
    #     ax.imshow(im_gt, cmap ='gray')
    #     ax.set_title("GT")
    #
    #     ax = plt.subplot("222")
    #     ax.imshow(im_bi, cmap ='gray')
    #     ax.set_title("Output(Bicubic)\nPSNR={}".format(psnr_bi))
    #
    #     ax = plt.subplot("223")
    #     ax.imshow(im_hr, cmap ='gray')
    #     ax.set_title("Output(Ours)\nPSNR={}".format(psnr_ours))
    #
    #     ax = plt.subplot("224")
    #     ax.imshow(im_another, cmap ='gray')
    #     ax.set_title("Output(SSR)\nPSNR={}".format(psnr_another))
    #     plt.tight_layout()     # https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle
    #     # plt.show()
    #     plt.savefig('{}/{}.png'.format(compare_dir, filename.split('.')[0]))
