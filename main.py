# coding=utf-8
from __future__ import print_function
import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from srresnet import Net
from torchvision import models
import torch.utils.model_zoo as model_zoo
from dataset import get_training_set, get_test_set
import torch.nn.functional as F
from torch.autograd import Variable
import dlib,time
from tensorboardX import SummaryWriter
import numpy as np
import scipy.misc as smi
from utils import PSNR

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.1, help="Clipping Gradients. Default=0.1")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
writer = SummaryWriter()
opt = parser.parse_args()

def main():
    # train_dir = "data/img_align_celeba"
    train_dir = "data/testset"
    crop_size = (128, 128)
    cpt_dir = "model/checkpoint_avrpool2"

    global opt, model, netContent
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
        
    print("===> Loading datasets")
    # train_set = DatasetFromHdf5("/path/to/your/hdf5/data/like/rgb_srresnet_x4.h5")
    train_set = get_training_set(train_dir, crop_size, upscale_factor=4, quality=40)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    if opt.vgg_loss:
        print('===> Loading VGG model')
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])
                
            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model()

    print("===> Building model")
    model = Net()
    # criterion = nn.MSELoss(size_average=False)
    criterion = CustomLoss(beta=0.5)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1   # opt.start_epoch 按照checkpoint文件计算
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            
    # optionally copy weights from a checkpoint
    if opt.pretrained:      # opt.start_epoch = 1
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
            
    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = StepLR(optimizer, step_size=opt.step, gamma=0.1, last_epoch=-1)
    print("===> Training")

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        scheduler.step()
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch, optimizer.param_groups[0]['lr'], cpt_dir)
        evaluate(model, opt.seed, "data/testset", crop_size, epoch=epoch, cpt_dir=cpt_dir)

def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))  # 设置学习率衰减规则，每opt.step个epoch 学习率减小10倍
    return lr    

def train(training_data_loader, optimizer, model, criterion, epoch):

    # lr = adjust_learning_rate(epoch-500-1)
    #
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr

    print("epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"])
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        start_time = time.time()
        input, target, lr_mask, hr_mask = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            lr_mask = lr_mask.cuda()
            hr_mask = hr_mask.cuda()

        output = model(input)
        loss = criterion(input, output, target, lr_mask, hr_mask)

        if opt.vgg_loss:
            content_input = netContent(output)
            content_target = netContent(target)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target)
        
        optimizer.zero_grad()

        if opt.vgg_loss:
            netContent.zero_grad()
            content_loss.backward(retain_variables=True)    # 如果设置vgg_loss，则单独对content_loss进行一次反向传播
        
        loss.backward()

        optimizer.step()
        elapsed_time = time.time() - start_time
        if iteration%100 == 0:

            if opt.vgg_loss:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f} Content_loss {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0], content_loss.data[0]))
            else:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f} time: {}".format(epoch, iteration, len(training_data_loader), loss.data[0], elapsed_time))
            writer.add_scalar('Loss', loss.data[0], iteration)

def save_checkpoint(model, epoch, lr, cpt_dir):
    model_out_path = "{}/base500_model_epoch_{}_lr_{}.pth".format(cpt_dir, epoch, lr)
    state = {"epoch": epoch, "lr":lr, "model": model}    # 保存为字典，分别记录epoch和model
    if not os.path.exists(cpt_dir):
        os.makedirs(cpt_dir)

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

def evaluate(model, seed, eval_path, crop_size, epoch, cpt_dir):
    testset = get_test_set(eval_path, crop_size, upscale_factor=4, quality=40)
    psnrs = []
    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    for i in range(0, testset.__len__()):
        im_l, im_gt, _, _ = testset.__getitem__(i)
        im_l = im_l.numpy().astype(np.float32)

        im_input = im_l.reshape(1, im_l.shape[0], im_l.shape[1], im_l.shape[2])
        im_input = Variable(torch.from_numpy(im_input).float())
        if opt.cuda:
            im_input = im_input.cuda()

        out = model(im_input)
        out = out.cpu()
        im_h = out.data[0].numpy().astype(np.float32)
        im_h = im_h*255.
        im_h[im_h<0] = 0
        im_h[im_h>255.] = 255.
        im_h = im_h.transpose(1,2,0).astype(np.uint8)
        im_hr_grey = np.array(smi.toimage(im_h).convert('L'))

        im_gt = im_gt.numpy().transpose(1, 2, 0)
        im_gt = (im_gt.astype(float) * 255.).astype(np.uint8)
        im_gt_grey = np.array(smi.toimage(im_gt).convert('L'))

        psnr_output = PSNR(im_hr_grey, im_gt_grey)
        psnrs.append(psnr_output)
    psnr_mean = np.array(psnrs).mean()
    writer.add_scalar("seed_{}/psnr_mean".format(seed),psnr_mean,epoch)

    with open("{}/mean_psnr.txt".format(cpt_dir), 'a') as f:
        f.write('epoch{}: mean psnr on eval set: {}\n'.format(epoch,psnr_mean))
    print("seed {} : mean psnr on eval set: {}".format(seed, psnr_mean))

class CustomLoss(nn.Module):
    def __init__(self, beta):
        super(CustomLoss, self).__init__()
        self.beta = beta

    def forward(self, input, output, target, lr_mask, hr_mask):
        # loss_1 = nn.MSELoss(size_average=False)
        output = hr_mask * output
        target = hr_mask * target
        loss_1 = F.mse_loss(output, target, size_average=False)
        # filter = Variable(torch.cuda.FloatTensor(3,3,1,1).fill_(1), requires_grad=False)
        # print filter, type(filter)
        # downsample = F.conv2d(target, filter, stride=4)
        # downsample = F.max_pool2d(target, 4, 4)
        downsample = lr_mask * F.avg_pool2d(target, 4, 4)
        input = lr_mask * input
        loss_2 = F.mse_loss(downsample, input)
        loss = loss_1 + self.beta * loss_2
        return loss

if __name__ == "__main__":
    main()
