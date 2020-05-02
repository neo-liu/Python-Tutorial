# author: Neo Liu
# data: 2020.4.9
# function: 处理光强分布，获取直接的Probe光谱
# 此包中包含的变量、函数是光纤分辨系统中需要的基础函数

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from IPython import display
from scipy.interpolate import interp1d

# 变量
resolution = 0  # 光谱分辨率
data = np.array([])  # 读取数据的中间变量
Data = list()  # 读取数据函数中的变量
Data_set = list()  # 读取的数据集（用于表示所有数据），使用于C_function.py中
Correlation = list()  # 光谱相关度，使用于C_function.py
T_matrix = list()  # 传输矩阵，使用于除C_function.py、2D_plot.py、3D_plot.py以外的所有代码中
boundary = np.array([])  # 单个散斑图样的数据尺度，用与数据整形
Intensity_Vector = np.array([])  # 特征谱线对应的光强矢量（K: 766.46nm, 769.9nm; Cu: 510.5nm,521.8nm）
Reconstruction_Spectral_Vector = np.array([])  # 重建光谱矢量，使用传输矩阵和特征谱线光强分布重建的光谱用此变量代替
miu = np.array([])  # 重建错误率


# 函数
plt = plt  # 绘图库
display = display  # 绘制矢量图函数
interp1d = interp1d  # 插值函数


# 读取数据维度（指单个散斑图长宽方向的数据个数），用来整形光强分布数据
def get_shape(name):
    i, j = 0, 0
    with open(str(name), 'r') as file:
        for num, line in enumerate(file):
            if num < 6:
                pass
            else:
                i += 1
                line = [i for i in line.split(' ')[1: -1]]
                j += len(line)
    raw, col = i, j//i
    print('获取数据维度 完成 . . . \n # # # # # # # # # # \n')
    return raw, col


# 于上述函数类似，应该合并在一起
def read_format(name):
    data = list()
    with open(str(name), 'r') as f:
        for line in f:
            line = [float(i) for i in line.split(' ')]
            data.append(line)
        print('获取数据格式 完成 . . . \n # # # # # # # # # # \n')
    return np.array((data[0][0], data[1][0]))


# 读取数据，可视情况读取单个或多组数据
def read_file(name, Data=None, shape=None, multi=True, num=None):

    with open(str(name), 'r') as f:
        data = np.zeros(shape=shape)
        for num1, line in enumerate(f):
            if num1 < 6:
                pass
            else:
                line_bak = [float(i)**2 for i in line.split(' ')[1: -1]]
                for num2, i in enumerate(line_bak):
                    data[num1 - 6][num2] = i
        if num and num % 20 == 0:
            print('读取' + str(num) + '组数据 完成 . . . \n # # # # # # # # # # \n')
    if not multi:
        return data
    elif multi:
        data = np.expand_dims(data, axis=0)
        if len(Data)==0:
            Data = data
        else:
            Data = np.append(Data, data, axis=0)
        return Data


def G(x1, x2, alpha=0.02):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha\
                             * np.exp(-((x1-x2) / alpha)**2 * np.log(2))