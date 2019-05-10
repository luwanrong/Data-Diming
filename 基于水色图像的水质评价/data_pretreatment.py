# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:12:05 2019

@author:luwanrong

E-mail:lwr6608@163.com
"""

import pandas as pd
import numpy as np
import matplotlib.image as img
import os

#图像分割
#打开图片路径，打开图片
imagesList = os.listdir('./data/images')

def img_feature(file_path):
    temp_array=np.zeros((1,9))
    img_info=img.imread(file_path)
    l_max = img_info.shape[0]//2+50
    l_min = img_info.shape[0]//2-50
    w_max = img_info.shape[1]//2+50
    w_min = img_info.shape[1]//2-50
    imgs = img_info[l_min:l_max, w_min:w_max, :].reshape(1, 10000, 3)
    #开始处理每一个通道
    for i in range(3):
        imgs_s = imgs[:, :, i]/256.0
        #通道1
        temp_array[0,i] = np.mean(imgs_s)
        
        #通道2
        temp_array[0,i+3] = np.sqrt(np.mean(np.square(imgs_s - temp_array[0,i])))
        
        #通道3
        temp_array[0,i+6] = np.cbrt(np.mean(np.power(imgs_s - temp_array[0,i],3)))
    
    return temp_array
    
temp = np.zeros((len(imagesList),10))      
for i,pic_name in enumerate(imagesList):
    temp[i,:9] = img_feature('./data/images/'+pic_name)
    temp[i,9]= int(pic_name.split('_')[0])
    
data = pd.DataFrame(temp,columns = ['R1','G1','B1','R2','G2','B2','R3','G3','B3','labels'])
data.to_csv('./data/modelfile.csv',index = 0)   