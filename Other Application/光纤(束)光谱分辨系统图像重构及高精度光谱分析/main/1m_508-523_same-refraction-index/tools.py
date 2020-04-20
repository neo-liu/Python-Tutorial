# author: Neo Liu
# data: 2020.4.9
# function: 处理光强分布，获取直接的Probe光谱
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from IPython import display
from scipy.interpolate import interp1d

plt = plt
display = display
interp1d = interp1d
resolution = 0
Data = list()
Correlation = list()
T_matrix = list()
boundary = list()
Intensity_Vector = list()
Reconstruction_Spectral_Vector = list()
miu = list()
Spectral_Vector = list()


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


def read_format(name):
    data = list()
    with open(str(name), 'r') as f:
        for line in f:
            line = [float(i) for i in line.split(' ')]
            data.append(line)
        print('获取数据格式 完成 . . . \n # # # # # # # # # # \n')
    return np.array((data[0][0], data[1][0]))


def read_file(name, shape=None, multi=True, num=None):

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
        Data.append(data)
        return Data


