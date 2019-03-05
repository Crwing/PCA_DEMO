# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as linA # 为了激活线性代数库mkl
from PIL import Image
import os,glob

def sim_distance(train,test):
    '''
    计算欧氏距离相似度
    :param train: 二维训练集
    :param test: 一维测试集
    :return: 该测试集到每一个训练集的欧氏距离
    '''
    return [np.linalg.norm(i - test) for i in train]

picture_path = os.getcwd()+'\\NEW_PIC\\'
array_list = []
loopindex = 0
for name in glob.glob(picture_path+'*.bmp'):
    # 读取每张图片并生成灰度（0-255）的一维序列 1*10304
    loopindex = loopindex + 1
    img = Image.open(name)
    print(loopindex.__str__() + '' + name)

    array_list.append(np.array(img).reshape((1,10304)))

mat = np.vstack((array_list)) # 将上述多个一维序列合并成矩阵 3*120000
mat_old = mat
mean_mat = np.mean(mat, axis=0)
#print(mean_mat)
mat = mat - mean_mat
P = np.dot(mat,mat.transpose()) # 计算P
v,d = np.linalg.eig(P) # 计算P的特征值和特征向量
sorted_indices = np.argsort(-v)
#print(v)
total_v = np.ndarray.sum(v)
cumsum = 0.0
index = 0
for i in v:
    index = index + 1
    cumsum = i + cumsum
    if  cumsum / total_v > 0.9:
        break
#print(index)
#print(sorted_indices)
k = index
d = d[:,0:k]

d= np.dot(mat.transpose(),d) # 计算Sigma的特征向量
train = np.dot(mat_old,d) # 计算训练集的主成分值

# 开始测试
test_pic = np.array(Image.open(picture_path + 's27_3.bmp')).reshape((1,10304))
result = sim_distance(train,np.dot(test_pic,d))

print(result.index(min(result)))

test_pic = np.array(Image.open(picture_path + 's26_10.bmp')).reshape((1,10304))
result = sim_distance(train,np.dot(test_pic,d))
print(result.index(min(result)))