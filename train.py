'''
@File    :   train.py
@Time    :   2021/02/25 09:33:31
@Author  :   Jack Wang
@Contact :   mrwanghongji@qq.com
@Address :   zhengzhou

'''

import torch                                               
import torch.nn as nn
import os
from time import time
from networks.dinknet import  DinkNet34
from networks.fcn8s import FCN8S
from networks.unet import Unet
from networks.unet_m import Unet_M
from networks.deeplabv3 import DeepLabv3_plus
from framework import MyFrame
from loss import dice_bce_loss
import torch.utils.data as Data
from data import MyData
import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


#上色的rgb值
back = [0,0,0]#黑色
stalk = [0 ,92,230]#蓝色
twig = [0,255,0]#绿色
grain = [128,128,0]#黄褐色
grain2 = [0,128,0]
whit = [255,255,255]
zi = [128,0,255]
COLOR_DICT = np.array([back,zi,twig,stalk,grain,grain2,whit])
num_class=5

#数据路径
images = r'E:\RandomTasks\Dlinknet\dataset\train\images'
mask = r'E:\RandomTasks\Dlinknet\dataset\train\labels'
voc_val = MyData(images,mask)

batchsize = 16#计算批次大小
train_load = Data.DataLoader(
    voc_val,
    batch_size=batchsize,
    shuffle=True)

NAME = 'DinkNet34_class8_xiaomai'#数据模型
modefiles = 'weights/'+NAME+'.th'
write = SummaryWriter('weights')#可视化
loss = nn.NLLLoss()
#loss = nn.CrossEntropyLoss()
solver = MyFrame(DinkNet34, loss, 0.0003)#网络，损失函数，以及学习率
if os.path.exists(modefiles):
    solver.load(modefiles)
    print('继续训练')

#灰度图上色代码
def IamColor(data):
    
    img_out = np.zeros(data.shape + (3,))
    for ii in range(num_class):
            img_out[data == ii,:] = COLOR_DICT[ii]#对应上色
    data = img_out / 255
    data = np.transpose(data,(0,3,1,2))

    return data


#训练网络
#可视化脚本tensorboard --logdir=weights --host=127.0.0.1
tic = time()
no_optim = 0
total_epoch = 20#训练次数
train_epoch_best_loss = 100.
for epoch in tqdm.tqdm(range(1, total_epoch + 1)):
    train_epoch_loss = 0
    for i,(img,mask) in enumerate(train_load):
        #总循环次数
        allstep=epoch*len(train_load)+i+1
        solver.set_input(img, mask)
        #网络训练，返回loss和网络输出
        train_loss,netout = solver.optimize()
        
        # #可视化训练数据
        # img_x = vutils.make_grid(img,nrow=4,normalize=True)
        # write.add_image('train_images',img_x,allstep)
        
        # #可视化标签
        # mask_pic=IamColor(mask)
        # mask_pic = torch.from_numpy(mask_pic)
        # mask_pic = vutils.make_grid(mask_pic,nrow=4,normalize=True)
        # write.add_image('label_images',mask_pic,allstep)
        
        # #可视化网络输出
        # pre = torch.argmax(netout.cpu(),1)
        # img_out = np.zeros(pre.shape + (3,))
        # for ii in range(num_class):
        #         img_out[pre == ii,:] = COLOR_DICT[ii]#对应上色
        # pre = img_out / 255
        # pre = np.transpose(pre,(0,3,1,2))#变成b c h w
        # pre = torch.from_numpy(pre)
        # img_out = vutils.make_grid(pre,nrow=4,normalize=True)#必须是tensor
        # write.add_image('predict_out',img_out,allstep)#必须是三个通道的

        #可视化损失函数输出
        train_epoch_loss += train_loss#所有的loss和
        write.add_scalar('train_loss',train_loss,allstep)
        # #可视化网络参数直方图感觉影响速度
        # for name,param in solver.net.named_parameters():
        #     write.add_histogram(name,param.data.cpu().numpy(),allstep)
    train_epoch_loss /= len(train_load)#平均loss
    print('********')
    print('epoch:',epoch,'time:',int(time()-tic)/60)
    print('train_loss:',train_epoch_loss)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss #保留结果
        solver.save(modefiles)
    if no_optim > 6:
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True)


