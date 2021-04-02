"""
加载自己的数据集
"""
import glob
import torch.utils.data as Data
from torchvision import transforms
import torch
import skimage.io
import os
import numpy as np
from skimage import measure
import numpy as np
import random
import cv2
from tqdm import tqdm
import shutil
from collections import Counter
import pandas as pd
import time
import numpy as np
import skimage.io
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_holes
import gdal

def post_proc(img, min_size=0, area=50):
    """
    删除小斑点，补全大漏洞
    """
    ind = remove_small_holes(label(img), min_size=min_size, connectivity=img.ndim)
    
    img = ind.astype(np.uint8)
    lab_arr = label(img)
    lab_atr = regionprops(lab_arr)
    
    def fun(atr):
        if atr.area <= area:
            min_row, min_col, max_row, max_col = atr.bbox
            t = lab_arr[min_row:max_row, min_col:max_col]
            t[t==atr.label] = 0

    list(map(fun, lab_atr))
    ind = lab_arr > 0
    sub = ind.astype(np.uint8)
    
    return sub


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

class MyData(Data.Dataset):

    def __init__(self, imagepath, maskpath):
        super(MyData, self).__init__()
        self.imagepath = imagepath
        self.maskpath = maskpath
        self.imagelist = glob.glob(os.path.join(imagepath, "*.tif"))
        self.masklist = glob.glob(os.path.join(maskpath, "*.png"))

    # 归一化

    def TransForm(self, image, mask):
        image_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])
        tens_image = image_t(image.astype(np.float32))  # 转化image为tensor
        tens_mask = torch.from_numpy(mask)
        return tens_image, tens_mask

    # 调用对象P[k]就会执行这个方法
    def __getitem__(self, index):
        oneimg = self.imagelist[index]  # 获取路径
        onemask = self.masklist[index]
        img = skimage.io.imread(oneimg)
        if img.shape[-1]>3:
            img = ChangeChen(skimage.io.imread(oneimg))  # 读取图片
        mask = skimage.io.imread(onemask).astype(np.int64)

        return self.TransForm(img, mask)

    def __len__(self):
        return len(self.imagelist)


class data_tools():
    

    def check_range(self,data, valid, n=3):
        """
        :param data: 数组数据
        :param valid: 无效值数值
        :return: 有效数值是否符合7/8
        """
        #data_v = sum(data.flatten() == valid)  # 计算影像中的0值

        A = Counter(data.flatten())
        data_v = A[valid]
        all_data = data.shape[0] * data.shape[1] * n
        sign_v = float(data_v / all_data)
        if sign_v < 0.01:  # mask覆盖范围和影像较多
            return True
        else:
            return False

    def tiftopng(self,path):
        list = os.listdir(path)
        for x in tqdm.tqdm(list):
            one = path + x
            a = skimage.io.imread(one)
            new = path + x.replace('tif', 'png')
            skimage.io.imsave(new, a)

            # os.rename(one, new)
    #按照大小删除
    def del_creat_list(self,pictype,masktype,min_file_size,label_rootdir):
        list = os.listdir(label_rootdir)

        temp_del = []
        for item in tqdm(list):
            file_path = label_rootdir + item
            # if '.tif' in file_path:
            #     os.remove(file_path)
            file_size = os.path.getsize(file_path)
            #img = skimage.io.imread(file_path)
            #full = self.check_range(img,0,3)
            # print(file_size)
            if file_size < min_file_size * 256:# or not full or img.shape[0]!=img.shape[1]:
                c = item  # .replace('tif','png')
                temp_del.append(item)
                more=file_path.replace('labels','images').replace('png',pictype)
                os.remove(file_path)
                os.remove(more)

        #return temp_del  # 文件名

    def del_list(self,list1):
        src_rootdir = r'E:\segmentation\dataset2\train\images/'
        a = os.listdir(src_rootdir)
        # for x in tqdm.tqdm(a):
        #     one = src_rootdir+x
        #     os.remove(one)
        for item in list1:
            file_path = src_rootdir + item.replace('png','tif')
            os.remove(file_path)

    # 两个文件夹内容保持一致
    def samepic(self):
        # loss_path = r'E:\segmentation\dataset\train\labels/'
        # more_path = r'E:\segmentation\dataset\train\images/'
        more_path=r"G:\Opendata\TianChiData\data\train\labels/"
        loss_path = r"G:\Opendata\TianChiData\data\train\images/"
        allpic = glob.glob(os.path.join(loss_path,"*.jpg"))
        morepic= glob.glob(os.path.join(more_path,"*.png"))
        for x in tqdm(morepic):#遍历多的文件
            ok = x.replace('png','jpg').replace('labels','images')#和少的对比
            if ok not in allpic:
                os.remove(x)

    def copyall(self,src_rootdir):#复制新的文件
        # src_rootdir = r'F:\cmm\TiCi\pic/'

        a = os.listdir(src_rootdir)
        for x in tqdm(a):

            one = os.path.join(src_rootdir.replace('labels','images'),x.replace('png','tif'))
            new = one.replace('images','new')
            shutil.copy(one,new)
    def rename(self,src_rootdir,sign):
        # src_rootdir = r'F:\cmm\TiCi\pic/'

        a = glob.glob(os.path.join(src_rootdir,"*.%s"%sign))
        for one in a:

            new = one.replace('train','images')
            shutil.copy(one,new)


size = 256
# 随机窗口采样
def change_train_dataset(pic_path,mask_path,delsize,data_type='tif',mask_type = 'png',image_num = 1000,
                           train_image_path='dataset/train/images/',
                           train_label_path='dataset/train/labels/'):
    '''
    该函数用来生成训练集，切图方法为随机切图采样
    :param image_num: 生成样本的个数
    :param train_image_path: 切图保存样本的地址
    :param train_label_path: 切图保存标签的地址
    :return:
    '''
    if not os.path.exists('dataset/train/images'): os.makedirs('dataset/train/images')
    if not os.path.exists('dataset/train/labels'):os.makedirs('dataset/train/labels')
    images_path = []
    labels_path = []
    # 用来记录所有的子图的数目
    g_count = 1
    allmask = glob.glob(os.path.join(mask_path,"*.%s"%mask_type))
    for one in allmask:
        labels_path.append(one)
        d,n=os.path.split(one)
        one_pic = os.path.join(pic_path,n.replace(mask_type,data_type))
        images_path.append(one_pic)

    for i in tqdm(range(len(images_path))):
        count = 0
        image = skimage.io.imread(images_path[i])#读取pic
        label = skimage.io.imread(labels_path[i],-1)#读取mask,灰度图方式
        X_height, X_width = image.shape[0], image.shape[1]#获取图片长
        if X_height>size and X_width>size:
            while count <image_num+1:
                random_width = random.randint(0, X_width - size - 1)#随机生成宽
                random_height = random.randint(0, X_height - size - 1)#随机生成长
                image_ogi = image[random_height: random_height + size, random_width: random_width + size,:]#裁剪
                label_ogi = label[random_height: random_height + size, random_width: random_width + size]#裁剪

                imagepath=train_image_path+'%05d1.%s' % (g_count,data_type)
                labelpath=train_label_path+'%05d1.png' % (g_count)
                skimage.io.imsave(imagepath, image_ogi)#保存图像
                skimage.io.imsave(labelpath, label_ogi)#np.array(label_ogi,dtype=np.uint8))
                count += 1
                g_count += 1
    a = data_tools()
    a.del_creat_list(data_type,mask_type, delsize, train_label_path)



class TCData(object):
        # 将图片编码为rle格式
    def rle_encode(self,im):
        '''
        im: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = im.flatten(order = 'F')
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    # 将rle格式进行解码为图片
    def rle_decode(self,mask_rle, shape=(512, 512)):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background

        '''
        try:
            s = mask_rle.split()
            starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
            starts -= 1
            ends = starts + lengths
            img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
            for lo, hi in zip(starts, ends):
                img[lo:hi] = 1
            return img.reshape(shape, order='F')
        except:
            img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
            
            return img.reshape(shape, order='F')

class BoDuanHeCheng(object):
    """
    提取需要的波段，波段合并
    """

    def img_write(self,img_path,img_array, img_proj, img_geotrans,datatype = gdal.GDT_UInt16):
        '''
    输入，数据信息，写入影像文件
    :param img_path: 
    :param img_array: 
    :param img_proj: 
    :param img_geotrans: 
    :param datatype: 
    :return: 
    '''
        assert  len(img_array.shape)==3
        driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
        dataset = driver.Create(img_path, img_array.shape[2], img_array.shape[1], img_array.shape[0], datatype,
                                options = ['COMPRESS=LZW' ])
        # other options: 'BigTIFF=YES', 'TILED=YES',"INTERLEAVE=PIXEL",'COMPRESS=PACKBITS'
    
        dataset.SetGeoTransform(img_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(img_proj)  # 写入投影
        for index,band in enumerate(img_array):
            dataset.GetRasterBand(index + 1).WriteArray(band)

    def ReadTIF(self,path):
        dataset=gdal.Open(path)
        img_proj = dataset.GetProjection()  # 地图投影信息
        img_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize
        im_data = dataset.ReadAsArray(0,0,im_width,im_height)
        return im_data,img_proj,img_geotrans
    def excet(self):
        imglist = glob.glob(os.path.join(r'Z:\张铮豪\河南省\2021年2月份小麦\影像','*.tif'))
        for patt in imglist:
            data,a,b=ReadTIF(patt)
            nine = data[8]
            seven = data[6]
            three = data[2]
            End = np.concatenate((
                nine[None,:,:],
                seven[None,:,:],
                three[None,:,:]),axis=0)
            save_path = patt.split('.')[0]+'3BD.tif'
            print(save_path)
            self.img_write(save_path,End,a,b)
            print(End.shape)
            

if __name__ == '__main__':
    new_data = []
    tool = TCData()
    train_mask = pd.read_csv(r'E:\Data\TianChi\test_a_samplesubmit.csv',sep='\t', names=['name', 'mask'])
    # 读取第一张图，并将对于的rle解码为mask矩阵
    for i in tqdm(range(len(train_mask['name']))):
        picname = train_mask['name'].iloc[i][:-4]+'.png'
        img = skimage.io.imread(r'E:\RandomTasks\Dlinknet\submits\zxnet/'+ picname)
        img = post_proc(img)
        mask = tool.rle_encode(img)
        # if len(mask)>32000:
        #     mask = mask[0:31999]
        #     print(picname)
        cont = train_mask['name'].iloc[i].split('/')[-1]
        new_data.append([cont,mask])
        # new_data.append(ok)
    df = pd.DataFrame(new_data)
    df.to_csv('terminator.csv',index=False,header=False)

        
        # savepath=os.path.join(r'E:\Data\TianChi\label',picname.split('.')[0]+'.png')
        # skimage.io.imsave(savepath,mask)

# # 结果为True