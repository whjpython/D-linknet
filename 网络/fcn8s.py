
import torch.nn as nn
from torchvision.models import vgg19


class FCN8S(nn.Module):
    def __init__(self):
        super().__init__()

        mode_vgg19 = vgg19(pretrained=True).cuda()
        self.base_model = mode_vgg19.features
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512,512,3,2,1,1,1)  # 尺寸增加一倍16
        self.bn1 =nn.BatchNorm2d(512)  # 归一化，网络结构稳定
        self.deconv2= nn.ConvTranspose2d(512,256,3,2,1,1,1)  # 32
        self.bn2 =nn.BatchNorm2d(256)  # 归一化，网络结构稳定
        self.deconv3= nn.ConvTranspose2d(256,128,3,2,1,1,1)  # 64
        self.bn3 =nn.BatchNorm2d(128)  # 归一化，网络结构稳定
        self.deconv4= nn.ConvTranspose2d(128,64,3,2,1,1,1)  # 128
        self.bn4 =nn.BatchNorm2d(64)  # 归一化，网络结构稳
        self.deconv5= nn.ConvTranspose2d(64,32,3,2,1,1,1)  # 256
        self.bn5 =nn.BatchNorm2d(32)  # 归一化，网络结构稳定
        self.classifier = nn.Conv2d(32,11,1)
        self.layer = {
            '4' :'maxpool_1' ,'9' :'maxpool_2' ,'18' :'maxpool_3',
            '27' :'maxpool_4' ,'36' :'maxpool_5'
        }
    def forward(self, x):
        output ={}
        for name ,layer in self.base_model._modules.items()  :  # 层数索引.层 
            x=layer(x)
            if name in self.layer:
                output[self.layer[name]]=x
        x5 = output['maxpool_5']  # 8
        x4 = output['maxpool_4']  # 16
        x3 = output['maxpool_3']  # 32
        score=self.relu(self.deconv1(x5))# 16,512
        score=self.bn1(score+x4)
        score=self.relu(self.deconv2 (score))  # 32,256
        score=self.bn2(score+x3)
        score=self.bn3(self.relu(self.deconv3(score)))  # 64
        score=self.bn4(self.relu(self.deconv4(score)))  # 128
        score=self.bn5(self.relu(self.deconv5(score)))  # 256
        score=self.classifier(score)

        return score