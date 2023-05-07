#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path as osp
import random

import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# In[3]:


def make_datapath_list(rootpath):
    """
    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        返回保存数据路径的列表
    """

    # 创建图像文件与标注文件的路径模板
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    # 分别取得训练和验证用的文件ID
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')

    # 创建训练惧的图像文件与标注文件的路径列表
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  
        img_path = (imgpath_template % file_id) 
        anno_path = (annopath_template % file_id) 
        train_img_list.append(img_path)  
        train_anno_list.append(anno_path)  


    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)  
        anno_path = (annopath_template % file_id)
        val_img_list.append(img_path) 
        val_anno_list.append(anno_path) 

    return train_img_list, train_anno_list, val_img_list, val_anno_list


# In[4]:


os.chdir('../Downloads/pytorch_advanced-master/2_objectdetection/')


# In[5]:


rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    rootpath)
#val_img_list


# In[6]:


# 对XML格式进行正规处理
class Anno_xml2list(object):
    """
    classes : 对XML正规处理之后保存到一个列表中
    """

    def __init__(self, classes):

        self.classes = classes

    def __call__(self, xml_path, width, height):
        """
        Returns
        -------
        ret : [[xmin, ymin, xmax, ymax, label_ind], ... ]
            保存的列表的格式
        """

        # 保存到这个列表
        ret = []

        # XML读取
        xml = ET.parse(xml_path).getroot()

        # 遍历XML中所有名为object的元素
        for obj in xml.iter('object'):

            # 将标注中的检测难度为difficult的剔除
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # 保存每个物体的标注信息的列表
            bndbox = []

            name = obj.find('name').text.lower().strip()  # 物体名
            bbox = obj.find('bndbox')  # 获取包围盒的信息

            # 将标注的坐标进行归一化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # VOC的原点是（1，1）所以要减掉1让原点为（0，0）
                cur_pixel = int(bbox.find(pt).text) - 1

                # 对宽度和高度正规化
                if pt == 'xmin' or pt == 'xmax':  # x方向用宽度除
                    cur_pixel /= width
                else:  # y方向用高度除
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            # 获取分类物体的名字的下标并且添加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

      
            ret += [bndbox]

        return np.array(ret)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


# In[7]:


# 确认执行结果
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

transform_anno = Anno_xml2list(voc_classes)
#transform_anno.classes


# In[8]:


#使用opencv读取图像
ind = 1
image_file_path = val_img_list[ind]
img = cv2.imread(image_file_path)
image_file_path


# In[9]:


height, width, channels = img.shape 

transform_anno(val_anno_list[ind], width, height)


# In[10]:


from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans


class DataTransform(): # 数据增强
   

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),  # int转化为32为的float
                ToAbsoluteCoords(),  # 返回标准化后的标注数据
                PhotometricDistort(),  # 随机的调整图像的色调
                Expand(color_mean),  # 拓展图像的画布尺寸
                RandomSampleCrop(),  # 随机的截取图像的部分内容
                RandomMirror(),  # 对图像进行翻转
                ToPercentCoords(),  # 对图像进行归一化
                Resize(input_size),  
                SubtractMeans(color_mean)  # 减去BGR的颜色平均值
            ]),
            'val': Compose([
                ConvertFromInts(),  
                Resize(input_size), 
                SubtractMeans(color_mean) 
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


# In[11]:


# 1. 读取图像
image_file_path = train_img_list[0]
img = cv2.imread(image_file_path)  
height, width, channels = img.shape  

# 2. 将标注放入列表中
transform_anno = Anno_xml2list(voc_classes)
anno_list = transform_anno(train_anno_list[0], width, height)

# 3. 显示原图像
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))# cvtcolor是从BGR转化为RGB的方法
plt.show()

# 4. 创建预处理类
color_mean = (104, 117, 123) 
input_size = 300 
transform = DataTransform(input_size, color_mean)

# 5. 显示train图像
phase = "train"
img_transformed, boxes, labels = transform(
    img, phase, anno_list[:, :4], anno_list[:, 4])
plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
plt.show()


# 6. 显示val图像 
phase = "val"
img_transformed, boxes, labels = transform(
    img, phase, anno_list[:, :4], anno_list[:, 4])
plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
plt.show()


# In[13]:


img.shape


# In[12]:


class VOCDataset(data.Dataset):
    

    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform 
        self.transform_anno = transform_anno  

    def __len__(self):
       
        return len(self.img_list)

    def __getitem__(self, index):
        
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):

        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path) 
        height, width, channels = img.shape 

        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4]
        )

        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width


# In[19]:


color_mean = (104, 117, 123) 
input_size = 300 

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

#val_dataset.__getitem__(1)


# In[20]:


def od_collate_fn(batch):
    """
    这里实现DataLoader的方法有一点不一样，这里的labels是gt由横纵坐标和label组成
    我们总不能保证锚框中总是只有一个物品出现，如果有两个那么labels就是(2,5)
    有3个就是(3,5)
    所以为了能实现这种不同的Dataloader，有必要对collate_fn进行重新定义
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  #sample[0]是图像
        targets.append(torch.FloatTensor(sample[1]))  # sample[1]是标注gt

    # imgs是小批量大小的列表
    # 原先torch.Size([3, 300, 300])
    # 现在torch.Size([batch_num, 3, 300, 300])
    imgs = torch.stack(imgs, dim=0)

    # target是标注数据的正确答案的列表
    # 列表的大小与批次的大小一样
    # 列表target的元素为[n , 5]
    # n对于每幅图像都是不同的 ， 表示每幅图像中所含有的物体数量
    # 这个5就是我们的gt -> [xmin, ymin, xmax, ymax, class_index] 

    return imgs, targets

