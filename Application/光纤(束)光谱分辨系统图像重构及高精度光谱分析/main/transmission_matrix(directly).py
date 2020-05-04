# author: Neo Liu
# data: 2020.4.9
# function: 处理光强分布，获取直接的Probe光谱
from tools import *

# 定义超参数，用来控制整个程序的参数，可以根据不同情况灵活控制
name1 = './data format.txt'  # 此为光强分布的数据格式
name2 = './output_510.5.fld'  # 此为两个Probe谱线的光强分布数据
name3 = './output_521.8.fld'
start, end, step = 508.01, 523.01, 0.2  # 此为波长参数
sample_step = 30  # 此为光强分布的抽样间隔
threshold = 0.01  # 此为重建精确光谱时使用的阈值，见61 line

# 读取光强分布的数据格式
boundary = read_format(name1)
XY_num_diff = int(boundary[1] - boundary[0])

# 获取光强分布数据的维度，增强代码鲁棒性
shape = list(get_shape(name2))
if shape[0] / boundary[0] == 1.0 and shape[1] / boundary[1] == 1.0:
    print('数据维度合理\n # # # # # # # # # #\n')
else:
    raise Exception('数据维度不合理\n # # # # # # # # # #\n')

# 读取name1, name2中的光强数据并抽样， 如果只使用一个数据则可以注释掉包含 *2 的部分
Intensity_Vector1 = read_file(name2, Data=None, shape=shape, multi=False)
Intensity_Vector2 = read_file(name3, Data=None, shape=shape, multi=False)
Intensity_Vector = (Intensity_Vector1 + Intensity_Vector2)[:, int(
    XY_num_diff/2):int(boundary[1] - XY_num_diff/2)].reshape(-1, 1)[:: sample_step]
# Intensity_Vector = Intensity_Vector1[:, int(XY_num_diff/2):int(
#     boundary[1] - XY_num_diff/2)].reshape(-1, 1)[:: sample_step]

# 读取标定光强数据并抽样
for num, i in enumerate(np.arange(start, end, step)):
    data = read_file('./output_' + str(round(i, 2)) + '.fld', data, shape=shape, multi=True, num=num)
for num, i in enumerate(data):
    i = i[:, int(XY_num_diff/2):int(boundary[1] - XY_num_diff/2)].reshape(1, -1)
    T_matrix.append(i[0, ::sample_step])

# 计算传输矩阵的伪逆矩阵（包含了奇异值分解）
T_matrix = np.array(T_matrix).T
T_matrix_inv = np.linalg.pinv(T_matrix)

# 计算还原光谱
# *1 代表完整的还原光谱（不包含Probe的精确值，只包含Probe值的相邻谱线）
# *2 代表Probe的精确还原值
Spectral_Vector1 = np.array(T_matrix_inv.dot(Intensity_Vector))
Spectral_Vector2 = list()
# 因为还原光谱中的数值代表与Probe值的相似程度，所以还原光谱中的负值没有实际意义，进行了抛除操作
for num, i in enumerate(Spectral_Vector1):
    if i < 0:
        Spectral_Vector1[num] = 0
# 定义还原光谱绘图的横坐标（波长坐标）
Spectral_plot1 = np.arange(0, end-start, step) + start
Spectral_plot2 = list()
# 还原Probe精确值，将’过分‘小的值抛除掉
for num in range(len(Spectral_Vector1)):
    a, b = 0, 0
    if num != 74:
        a, b = Spectral_Vector1[num], Spectral_Vector1[num+1]
    elif num == 74:
        print('光谱还原 完成 . . . \n # # # # # # # # # # \n')
    if a > threshold and b > threshold:  # 进行筛选，只有连续两个值均大于0.01才可继续执行
        # 将a, b值得关系近似为线性关系
        Spectral_plot2.append((Spectral_plot1[num] + step * b / (a + b))[0])
        if a / b > 5:
            Spectral_Vector2.append(a[0])
        else:
            Spectral_Vector2.append(((a + b) / 2)[0])
    else:
        Spectral_plot2.append(Spectral_plot1[num])
        Spectral_Vector2.append(0)

# 绘制精确还原光谱
plt.figure(0)
plt.stem(Spectral_plot2, Spectral_Vector2, use_line_collection=True, linefmt='--', basefmt='b--')
for num, i in enumerate(Spectral_Vector2):
    i = round(i, 2)
    j = round(Spectral_plot2[num], 4)
    if Spectral_Vector2[num] >= 0.1:
        plt.annotate((j, i), xy=(Spectral_plot2[num], i),
                     xytext=(Spectral_plot2[num], i))
fun_values = list()
x = np.linspace(start, end, 10000)
for num1, num2 in zip(Spectral_plot2, Spectral_Vector2):
    y = G(x, num1)
    fun_values.append(y/max(y)*num2)
y = 0
for i in fun_values:
    y += i
plt.plot(x, y, color='r', ls='-')
# 绘制完整还原光谱
plt.figure(1)
plt.stem(Spectral_plot1, Spectral_Vector1, use_line_collection=True, linefmt='--', basefmt='b--')
for num, i in enumerate(Spectral_Vector1):
    i = round(i.tolist()[0], 2)
    if Spectral_Vector1[num] >= 0.1:
        plt.annotate((Spectral_plot1[num], i), xy=(Spectral_plot1[num], i),
                     xytext=(Spectral_plot1[num], i))
fun_values = list()
x = np.linspace(start, end, 10000)
for num1, num2 in zip(Spectral_plot1, Spectral_Vector1):
    y = G(x, num1)
    fun_values.append(y/max(y)*num2)
y = 0
for i in fun_values:
    y += i
plt.plot(x, y, color='r', ls='-')
plt.grid(), plt.xlabel('wavelength/nm', fontsize=15), plt.ylabel('Correlation', fontsize=15)
plt.xlim(508.0, 523.0)
plt.show()


