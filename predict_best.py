import skimage.io
import numpy as np
import os
from osgeo.gdalconst import *
from osgeo import gdal
import tqdm
import time
import glob
import torch
from networks.dinknet import  DinkNet34
from networks.deeplabv3 import DeepLabv3_plus
from torchvision import transforms
from networks.fcn8s import FCN8S
from networks.unet import Unet
from networks.unet_m import Unet_M
from data import post_proc

BATCHSIZE_PER_CARD = 4
back = [0,0,0]
stalk = [0 ,92,230]
twig = [0,255,0]
grain = [128,128,0]
grain2 = [0,128,0]
COLOR_DICT = np.array([back,stalk, twig, grain,grain2])#上色代码只有4类


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def predict_x(self,img):#这里是一个数组
        self.net.eval()#不会改变权重值
        img = img.cuda()#归一化，放在gpu上
        maska = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)#4*5*256*256
        return maska
    def load(self, path):
        self.net.load_state_dict(torch.load(path))


def ChangeChen(image):
    """
    将4通道变为3通道，同时改变通道的顺序
    """
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    n = image[:,:,3]
    x = np.concatenate(
        (n[:,:,None],g[:,:,None],r[:,:,None]),
        axis=2)
    return x

#num_class = 4#网络识别类别
class P():
    def __init__(self,number):
        self.number = number

    def Test_read(self,testpic):

        image_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])
        tenstest_image = image_t(testpic.astype(np.float32))#3*256*256

        return tenstest_image

    def CreatTf(self,file_path_img,data,outpath):#原始文件，识别后的文件数组形式，新保存文件
        d,n = os.path.split(file_path_img)
        dataset = gdal.Open(file_path_img, GA_ReadOnly)#打开图片只读
        #data = gdal.Open(os.path.join(outpath,'gyey'+n))#打开标签图片
        #data_label = data.ReadAsArray(0, 0, data.RasterXSize, data.RasterYSize)#获取数据
        projinfo = dataset.GetProjection()#获取坐标系
        geotransform = dataset.GetGeoTransform()
        #band = dataset.RasterCount()
        format = "GTiff"
        driver = gdal.GetDriverByName(format)#数据格式
        name = n[:-4]+'_result'+'.tif'#输出文件名字
        dst_ds = driver.Create(os.path.join(outpath,name), dataset.RasterXSize, dataset.RasterYSize,
                                  1, gdal.GDT_Byte )#创建一个新的文件
        dst_ds.SetGeoTransform(geotransform)#投影
        dst_ds.SetProjection(projinfo)#坐标
        dst_ds.GetRasterBand(1).WriteArray(data)
        dst_ds.FlushCache()


    def make_prediction_img(self,x, target_size, batch_size, predict):  # 函数当做变量
        """
        滑动窗口预测图像。

        每次取target_size大小的图像预测，但只取中间的1/4，这样预测可以避免产生接缝。
        """
        # target window是正方形，target_size是边长
        quarter_target_size = target_size//4
        half_target_size = target_size // 2


        pad_width = (
            (quarter_target_size, target_size),  # 32,128是因为遍历不到最后一个值
            (quarter_target_size, target_size), # 32,128
            (0,0))#第三个维度扩展维度为0，所以是0,0

        # 只在前两维pad
        pad_x = np.pad(x, pad_width, 'constant', constant_values=0)  # 填充(x.shape[0]+160,x.shape[1]+160)
        pad_y = np.zeros(
            (pad_x.shape[0], pad_x.shape[1],8),
            dtype=np.float32)  # 32位浮点型
        def update_prediction_center(one_batch):
            """根据预测结果更新原图中的一个小窗口，只取预测结果正中间的1/4的区域"""
            wins = []  # 窗口
            for row_begin, row_end, col_begin, col_end in one_batch:
                win = pad_x[row_begin:row_end, col_begin:col_end, :]  # 每次裁剪数组这里引入数据
                win = self.Test_read(win)#转换数据,会自动改变数据维度
                win = torch.unsqueeze(win,0)  # 喂入数据的维度确定了喂入的数据要求是(n, 3，256,256)
                wins.append(win)
            x_window = np.concatenate(wins, 0)  # 一个批次的数据
            x_window = torch.from_numpy(x_window)
            y_window = predict(x_window)  # 预测一个窗格，返回结果需要一个一个批次的取出来
            
            for k in range(len(wins)):  # 获取窗口编号
                
                row_begin, row_end, col_begin, col_end = one_batch[k]  # 取出来一个索引
                if len(y_window.shape)>3:
                    pred = y_window[k, ...]  # 裁剪出来一个数组，取出来一个批次数据5*256*256  
                if len(y_window.shape)==3:
                    pred = y_window       
                pred = np.transpose(pred,(1,2,0))#互换
                #直接把结果保存到空矩阵中效果不好
                # pad_y[
                # row_begin:row_end,col_begin :col_end,:
                # ] = pred
      
                 # 把预测的结果放到建立的空矩阵中[32:96，32:96]
                y_window_center = pred[
                                  quarter_target_size:target_size - quarter_target_size,
                                  quarter_target_size:target_size - quarter_target_size,
                                  :]  # 只取预测结果中间区域减去边界32[32:96,32:96]

                pad_y[
                row_begin + quarter_target_size:row_end - quarter_target_size,
                col_begin + quarter_target_size:col_end - quarter_target_size,:
                ] = y_window_center  # 只取4/1

        # 每次移动半个窗格
        batchs = []
        batch = []
        for row_begin in range(0, pad_x.shape[0], half_target_size):  # 行中每次移动半个[0,x+160,64]
            for col_begin in range(0, pad_x.shape[1], half_target_size):  # 列中每次移动半个[0,x+160,64]
                row_end = row_begin + target_size  # 0+128
                col_end = col_begin + target_size  # 0+128
                if row_end <= pad_x.shape[0] and col_end <= pad_x.shape[1]:  # 范围不能超出图像的shape
                    batch.append((row_begin, row_end, col_begin, col_end))  # 取出来一部分列表[0,128,0,128]
                    if len(batch) == batch_size:  # 够一个批次的数据
                        batchs.append(batch)
                        batch = []
        if len(batch) > 0:
            batchs.append(batch)
            batch = []
        for bat in tqdm.tqdm(batchs, desc='Batch pred'):  # 添加一个批次的数据
            update_prediction_center(bat)  # bat只是一个裁剪边界坐标
        y = pad_y[quarter_target_size:quarter_target_size + x.shape[0],
            quarter_target_size:quarter_target_size + x.shape[1],
            :]  # 收缩切割为原来的尺寸
        return y  # 原图像的预测结果

    def main_p(self,allpath,outpath,fun):#模型，所有图片路径列表，输出图片路径
        print('执行预测...')
        #allpath = glob.glob(os.path.join(allpath, "*.tif"))
        for one_path in allpath:
            t0 = time.time()
            #pic = ChangeChen(skimage.io.imread(one_path))
            pic = skimage.io.imread(one_path)
            # pic = pic.astype(np.float32)
            #torch框架维度在前面
            #识别大小应该小于输入图片
            y_probs = self.make_prediction_img(
                pic, 128, 8,
                lambda xx: fun.predict_x(xx))  # 数据，目标大小，批次大小，返回每次识别的
            y_probs = np.argmax(y_probs,2)#最终结果
            d, n = os.path.split(one_path)
            # change = y_probs.astype(np.uint8)

            self.CreatTf(one_path, y_probs,outpath)  # 添加坐标系
            print(n)
            # img_out = np.zeros(y_probs.shape + (3,))
            # for i in range(self.number):
            #      img_out[y_probs == i, :] = COLOR_DICT[i]#对应上色
            # y_probs = img_out / 255
            #save_file=os.path.join(outpath,n[:-4]+'_init'+'.png')
            # save_file=os.path.join(outpath,n[:-4]+'.png')
            # skimage.io.imsave(save_file, change)
            # try:
            #     skimage.io.imsave(save_file, change)
            # except:
            #     y_probs=np.zeros((256,256))
            #     skimage.io.imsave(save_file, y_probs)
            #os.startfile(outpath)
        print('预测耗费时间: %0.2f(min).' % ((time.time() - t0) / 60))



if __name__ == '__main__':

    path = r'Z:\张铮豪\河南省\2021年2月份小麦\DeepLearning\T'
    source1 = r'G:\Opendata\deepglobe-road-dataset\valid/'  # 识别路径
    solver = TTAFrame(DinkNet34)  # 根据批次识别类
    solver.load('weights/DinkNet34_class8_xiaomai.th')  # 加载模型
    #target = 'submits/deeplabv3plus/'  # 输出文件位置
    target=r'Z:\张铮豪\河南省\2021年2月份小麦\DeepLearning\233'
    if not os.path.exists(target):
        os.mkdir(target)
    listpic = glob.glob(os.path.join(path, "*.tif"))
    a = P(8)
    a.main_p(listpic, target, solver)