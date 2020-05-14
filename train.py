import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import os
from time import time
from networks.dinknet import  DinkNet34
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
import tqdm

SHAPE = (256,256)#数据维度
ROOT = r'E:\segmentation\dataset3\train\images/'
ROOT = r'G:\Opendata\deepglobe-road-dataset\train/'
imagelist = filter(lambda x: x.find('tif')!=-1, os.listdir(ROOT))#确定数据
trainlist = list(map(lambda x: x[:-4], imagelist))#取前面的名字
NAME = 'roadnew_dink34'#数据模型
modefiles = 'weights/'+NAME+'.th'


solver = MyFrame(DinkNet34, dice_bce_loss, 1e-5)#网络，损失函数，以及学习率
if os.path.exists(modefiles):
    solver.load(modefiles)
    print('继续训练')
batchsize = 4#计算批次大小

dataset = ImageFolder(trainlist, ROOT)#读取数据,文件路径设置，启用数据增强，传入数据
#参数只是传入文件共同名字
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)#多进程
#
mylog = open('logs/'+NAME+'.log','w')#日志文件
tic = time()
no_optim = 0
total_epoch = 300#训练次数
train_epoch_best_loss = 100.#预期结果
for epoch in tqdm.tqdm(range(1, total_epoch + 1)):
    data_loader_iter = iter(data_loader)#迭代器
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()#优化器
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    #solver.save('weights/' + NAME + '.th')
    print('********')
    print('epoch:',epoch,'    time:',int(time()-tic))
    print('train_loss:',train_epoch_loss)
    print('SHAPE:',SHAPE)
    print('********')
    print('epoch:',epoch,'    time:',int(time()-tic))
    print('train_loss:',train_epoch_loss)
    
    if train_epoch_loss >= train_epoch_best_loss:#保留最好的loss
        no_optim += 1
    else:#小于最好的
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss #保留结果
        solver.save(modefiles)
    if no_optim > 6:
        print(mylog, 'early stop at %d epoch' % epoch)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
    mylog.flush()

print('Finish!')
mylog.close()