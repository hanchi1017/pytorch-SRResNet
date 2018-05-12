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
import csv

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

opt = parser.parse_args()
print(opt)

train_dir = "data/train"
test_dir = "data/test"
crop_size = (128, 128)
cpt_dir = "model/checkpoint_vgg_relu22"
beta = 1

if not os.path.exists(cpt_dir):
    os.makedirs(cpt_dir)
resultFile = open("{}/result.csv".format(cpt_dir), 'a')
fieldnames = ['Epoch', 'Iteration', 'AvgPSNR', 'AvgCustomLoss']
resultWriter = csv.DictWriter(resultFile, fieldnames=fieldnames)
resultWriter.writeheader()
writer = SummaryWriter()

def main():


    global opt, model, netContent


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
    test_set = get_training_set(test_dir, crop_size, upscale_factor=4, quality=40)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    if opt.vgg_loss:
        print('===> Loading VGG model')
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:4])    # 使用relu2_2
                
            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model()

    print("===> Building model")
    model = Net()

    mseLoss = nn.MSELoss(size_average=False)
    criterion = CustomLoss(beta)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda()
            mseLoss.cuda()

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
        train(training_data_loader, test_data_loader, optimizer, model, criterion, mseLoss, epoch, opt.seed, cpt_dir)
        save_checkpoint(model, epoch, optimizer.param_groups[0]['lr'], cpt_dir)

    writer.close()
    resultFile.close()

def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))  # 设置学习率衰减规则，每opt.step个epoch 学习率减小10倍
    return lr    

def train(training_data_loader, test_data_loader, optimizer, model, criterion, mseLoss, epoch, seed, cpt_dir):

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
            content_loss = mseLoss(content_input, content_target)
        
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
            evaluate(test_data_loader, model, criterion, opt.seed, epoch, (epoch-opt.start_epoch)*len(training_data_loader)+iteration)


def save_checkpoint(model, epoch, lr, cpt_dir):
    model_out_path = "{}/base500_model_epoch_{}_lr_{}.pth".format(cpt_dir, epoch, lr)
    state = {"epoch": epoch, "lr":lr, "model": model}    # 保存为字典，分别记录epoch和model
    if not os.path.exists(cpt_dir):
        os.makedirs(cpt_dir)

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

def evaluate(test_data_loader, model, criterion, seed, epoch, iteration):

    avg_psnr = 0
    avg_loss = 0
    for iterarion, batch in enumerate(test_data_loader, 1):
        input, target, lr_mask, hr_mask = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)
        MSELoss = nn.MSELoss()
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            lr_mask = lr_mask.cuda()
            hr_mask = hr_mask.cuda()
            MSELoss = MSELoss.cuda()

        output = model(input)
        loss = criterion(input, output, target, lr_mask, hr_mask)
        avg_loss += loss.data[0]

        mse = MSELoss(output, target)
        psnr = 10 * math.log10(1 / mse.data[0])
        avg_psnr += psnr

    writer.add_scalar("seed_{}/CustomLoss_mean".format(seed), avg_loss, epoch)
    writer.add_scalar("seed_{}/psnr_mean".format(seed), avg_psnr, epoch)
    resultWriter.writerow({'Epoch':epoch, 'Iteration':iteration, 'AvgPSNR':avg_psnr, 'AvgCustomLoss':avg_loss})

    if iteration%1000 == 0:
        print("seed {}: ===> Epoch: {}; Iteration: {}; Avg. CustomLoss: {:.4f}; Avg. PSNR: {:.4f} dB".format(seed,epoch,iteration,avg_loss / len(test_data_loader), avg_psnr / len(test_data_loader)))   # len(test_data_loader) 为batch总数

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
