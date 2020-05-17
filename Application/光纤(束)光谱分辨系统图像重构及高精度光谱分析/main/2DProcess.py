import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from tools import *  # 导入读取数据、计算需要使用到的函数包

###################################################################################################

# 定义参数（用来控制整个程序），可以根据不同需求灵活控制
name1 = './data format.txt'  # 此为数据格式文件
name2 = './output_755.0.fld'  # 此为标定谱线的光强分布数据之一，用于读取数据维度使用

# 此为波长覆盖参数（对应与数据表的横向维度），波长范围λ从755.0至780.0nm，波长间隔为0.2nm，共126组
start, end, step = 755.0, 780.0, 0.2  # 用于还原光谱

# 获取光强分布数据的维度，用于后边散斑图的整形和验证读取数据的正确性
shape = list(get_shape(name2))

# 每个循环中都读取一个λ取值的数据，从start取值到end
for num1, i in enumerate(np.arange(start, end, step)):
    data = read_file('./output_' + str(round(i, 4)) + '.fld', data, shape, multi=True, num=num1)

# 读取数据的格式信息，包含边界、网格
boundary = list()
with open('data format.txt', 'r') as f:
    for line in f:
        line = [float(i) for i in line.split(' ')]
        boundary.append(line)
boundary = np.array(boundary)

# 提取网格和边界数据
X_slice_num, Y_slice_num = int(boundary[0][0]), int(boundary[1][0])
X_Min, X_Max = boundary[0][1], boundary[0][2]
Y_Min, Y_Max = boundary[1][1], boundary[1][2]

Y = np.linspace(X_Min, X_Max, X_slice_num)
X = np.linspace(Y_Min, Y_Max, Y_slice_num)

data = [np.array(i).T for i in data]

display.set_matplotlib_formats('svg')
# 绘图
for num, i in enumerate(data):
    plt.figure(str(num), figsize=(7, 6), dpi=100)
    levels = MaxNLocator(nbins=200).tick_values(np.min(i), np.max(i))
    cm = plt.cm.get_cmap('jet')
    # 填充颜色，f即为filled
    plt.contourf(Y, X, i, levels=levels, cmap=cm)

    # 绘制色条
    cbar = plt.colorbar()
    cbar.set_label('Intensity', rotation=-90, va='bottom', fontsize=18)
    cbar.set_ticks(np.linspace(np.min(i), np.max(i), 10))

    # set the font size of colorbar
    cbar.ax.tick_params(labelsize=10)
    # 画等高线
    # plt.contour(Y, X, data)
    plt.savefig('./image/'+str(num)+'.png')
    plt.show()
