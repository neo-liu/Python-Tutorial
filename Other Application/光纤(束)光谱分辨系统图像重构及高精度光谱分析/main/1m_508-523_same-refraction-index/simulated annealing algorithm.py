# author: Neo Liu
# data: 2020.4.9
# function: 处理光强分布，获取直接的Probe光谱
from tools import *
import math


def aimf(i, t, s):
    result = np.power(i - t.dot(s), 2).sum()
    return result


# 定义超参数，用来控制整个程序的参数，可以根据不同情况灵活控制
name1 = './data format.txt'  # 此为光强分布的数据格式
name2 = './output_766.46.fld'  # 此为两个Probe谱线的光强分布数据
name3 = './output_769.9.fld'
wave_start, wave_end, wave_step = 755.0, 780.0, 0.2  # 此为波长参数
miu_start, miu_end, miu_step = 0, 1, 1e-2
sample_step = 30  # 此为光强分布的抽样间隔
threshold = 0.01  # 此为重建精确光谱时使用的阈值，见61 line

# 读取光强分布的数据格式
boundary = read_format(name1)
XY_num_diff = int(boundary[1] - boundary[0])

# 获取光强分布数据的维度，增强代码鲁棒性
shape = list(get_shape(name2))
if shape[0] / boundary[0] == 1.0 and shape[1] / boundary[1] == 1.0:
    print('数据维度合理 \n # # # # # # # # # # \n')
else:
    raise Exception('数据维度不合理\n # # # # # # # # # #\n')

# 读取name1, name2中的光强数据并抽样， 如果只使用一个数据则可以注释掉包含 *2 的部分
Intensity_Vector1 = read_file(name2, shape, multi=False)
# Intensity_Vector2 = read_file(name3, shape, multi=False)
# Intensity_Vector = (Intensity_Vector1 + Intensity_Vector2)[:, int(
#     XY_num_diff/2):int(boundary[1][0] - XY_num_diff/2)].reshape(-1, 1)[:: sample_step]
Intensity_Vector = Intensity_Vector1[:, int(XY_num_diff/2):int(
    boundary[1] - XY_num_diff/2)].reshape(-1, 1)[:: sample_step]  # 整形，抽样

# 读取标定光强数据并抽样
for num, i in enumerate(np.arange(wave_start, wave_end+0.1, wave_step)):
    T_matrix = read_file('./output_' + str(round(i, 4)) + '.fld', shape, multi=True, num=num)

for num, i in enumerate(T_matrix):
    i = i[:, int(XY_num_diff/2):int(boundary[1] - XY_num_diff/2)].reshape(1, -1)
    T_matrix[num] = i[0, ::sample_step]

# 此为K原子特征光谱
# Probe_Intensity_Vector = np.array([766.46,  769.9])
Probe_Intensity_Vector = np.array([766.46, ])

print('Finish reading data files . . . ')
print('# # # # # # # # # #\n')

###################################################################################################

T_matrix = np.array(T_matrix).T
# 奇异分解
U, D, VT = np.linalg.svd(T_matrix)
print('D:', D)
D = np.concatenate((np.diag(D), np.zeros((T_matrix.shape[0]-D.shape[0], D.shape[0]))), axis=0)
print('Finish calculating SVD . . . ')
print('# # # # # # # # # #\n')

###################################################################################################

# **此为截断核心，将特征值中过小的元素进行截断**
# 这段代码也可以放进tools中
loop = 1
for gap in np.arange(miu_start, miu_end, miu_step):
    print(str(loop) + 'th truncation loop . . .')
    loop += 1
    for num1, i in enumerate(D):
        # print('num1', num1, 'gap', gap)
        for num2, j in enumerate(i):
            # j为截断阈值，需要进行调整
            if j <= gap:
                D[num1][num2] = np.inf
        # 将D矩阵元素进行倒数操作
    D_pit = np.reciprocal(D)
    T_matrix_inv = np.dot(VT.T, D_pit.T.dot(U.T))
    # 求解光谱矢量
    Reconstruction_Spectral_Vector.append(T_matrix_inv.dot(Intensity_Vector))

Reconstruction_Spectral_Vector = np.array(Reconstruction_Spectral_Vector)

for num1, vector in enumerate(Reconstruction_Spectral_Vector):
    for num2, i in enumerate(vector):
        if i < 0:
            Reconstruction_Spectral_Vector[num1][num2] = 0

print('Finish reconstruction . . . ')
print('# # # # # # # # # #\n')

###################################################################################################

Spectral_plot = np.arange(0, wave_end - wave_start + 0.1 * wave_step, wave_step) + wave_start

Probe_Intensity_Vector_bak = Probe_Intensity_Vector

print('序号 | list1 | list2 | difference | miu')
for i in range(len(Probe_Intensity_Vector_bak)):
    for j in range(len(Spectral_plot)):
        diff = Probe_Intensity_Vector_bak[i] - Spectral_plot[j]
        if 0 <= diff < wave_step:
            pass

        else:
            print(j, '|', Probe_Intensity_Vector_bak[i], '|', Spectral_plot[j], '|', diff)
            Probe_Intensity_Vector = np.insert(Probe_Intensity_Vector, j, 0)

for num, i in enumerate(Probe_Intensity_Vector):
    if i != 0:
        Probe_Intensity_Vector[num] = 1

# 求解重建错误率
for i in Reconstruction_Spectral_Vector:
    diff_vector = Probe_Intensity_Vector - i
    miu.append(np.sqrt(np.power(diff_vector, 2).mean()) / (Probe_Intensity_Vector**2).mean())

###################################################################################################
# 绘制错误率曲线
plt.figure(0)
plt.plot(np.arange(miu_start, miu_end, miu_step), miu, 'r--')

# 寻找错误率最小值的索引
index = miu.index(min(np.array(miu)))
print('\nReconstruction error is %.4f' % miu[index])
print('\nFinish calculating reconstruction error . . . ')
print('# # # # # # # # # #\n')

###################################################################################################

# simulated annealing algorithm
T = 1000
T_min = 10
k = 50
E = 0
t = 0
loop = 1
while T >= T_min:
    print(str(loop) + 'th simulated annealing loop . . . ')
    loop += 1
    for i in range(k):
        # calculate y
        E = aimf(Intensity_Vector, T_matrix, Reconstruction_Spectral_Vector[index])
        # generate a new x in the neighboorhood of x by transform function
        Reconstruction_Spectral_Vector_new = Reconstruction_Spectral_Vector[index] \
                                             + np.random.uniform(low=min(Reconstruction_Spectral_Vector[index]),
                                                                 high=max(Reconstruction_Spectral_Vector[index])) * T
        if 0 < Reconstruction_Spectral_Vector_new.min() <= 0.05:
            ENew = aimf(Intensity_Vector, T_matrix, Reconstruction_Spectral_Vector_new)
            if ENew - E < 0:
                Reconstruction_Spectral_Vector[index] = Reconstruction_Spectral_Vector_new
            else:
                # metropolis principle
                p = math.exp(-(ENew - E) / T)
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    Reconstruction_Spectral_Vector[index] = Reconstruction_Spectral_Vector_new
    t += 1
    T = 1000 / (1 + t)
print('\nFinish simulated annealing algorithm . . . ')
print('# # # # # # # # # #\n')

###################################################################################################

# 绘制光谱分布点列图
plt.figure(1)
plt.stem(Spectral_plot, Reconstruction_Spectral_Vector[index], use_line_collection=True,
         linefmt='r--', basefmt='b--')

for num, i in enumerate(Reconstruction_Spectral_Vector[index]):  # 标明光谱向量的值
    i = round(i.tolist()[0], 2)
    if Reconstruction_Spectral_Vector[index][num] >= 0.1:
        plt.annotate((Spectral_plot[num], i), xy=(Spectral_plot[num], i),
                     xytext=(Spectral_plot[num], i))
plt.grid(), plt.xlabel('wavelength/nm', fontsize=15), plt.ylabel('Correlation', fontsize=15)
print('Plotting figures . . . \n# # # # # # # # # # \n\nThe End')

plt.show()
