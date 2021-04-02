import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
from networks.utils import *
class MyFrame():
    """
    一些参数函数
    """
    def __init__(self, net,loss,lr=0.003):#loss是一个函数
        self.net = net().cuda()#GPUmodel = Unet(3, 3, convblock=SplAtBlock)
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))#多GPU训练
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)#优化器
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.old_lr = lr
        self.loss = loss

    def set_input(self, img_batch, mask_batch=None, img_id=None):#喂数据
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
        
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self):#添加变量把数据放到GPU上
        self.img = self.img.cuda()
        if self.mask is not None:
            self.mask = self.mask.cuda()
        
    def optimize(self):#这里将图片中的
        self.forward()#调用数据
        self.optimizer.zero_grad()#初始化网络参数,对参数进行清空
        pred_out = self.net.forward(self.img)#网络中输入数据
        pred = F.log_softmax(pred_out, dim=1)
        loss = self.loss(pred,self.mask)#计算误差
        loss.backward()#反向传播计算
        self.optimizer.step()#更新参数
        return loss.item(),pred_out#返回loss数值,和网络输出

    def save(self, path):
        torch.save(self.net.state_dict(), path)#保存模型
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))#加载模型
    
    def update_lr(self, new_lr,factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        self.old_lr = new_lr
