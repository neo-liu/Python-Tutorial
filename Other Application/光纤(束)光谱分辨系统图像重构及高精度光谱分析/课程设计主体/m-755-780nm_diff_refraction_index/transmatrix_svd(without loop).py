# author: Neo Liu
# data: 2020.4.9
# function: 处理光强分布，获取直接的Probe光谱
from tools import *

# 定义超参数，用来控制整个程序的参数，可以根据不同情况灵活控制
name1 = './data format.txt'  # 此为光强分布的数据格式
name2 = './output_766.46.fld'  # 此为两个Probe谱线的光强分布数据
name3 = './output_769.9.fld'
start, end, step = 755.0, 780.0, 0.2  # 此为波长参数
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

###################################################################################################

# 读取name1, name2中的光强数据并抽样， 如果只使用一个数据则可以注释掉包含 *2 的部分
Intensity_Vector1 = read_file(name2, shape, multi=False)
# Intensity_Vector2 = read_file(name3, shape, multi=False)
# Intensity_Vector = (Intensity_Vector1 + Intensity_Vector2)[:, int(
#     XY_num_diff/2):int(boundary[1][0] - XY_num_diff/2)].reshape(-1, 1)[:: sample_step]
Intensity_Vector = Intensity_Vector1[:, int(XY_num_diff/2):int(
    boundary[1] - XY_num_diff/2)].reshape(-1, 1)[:: sample_step]  # 整形，抽样

# 读取标定光强数据并抽样
for num, i in enumerate(np.arange(start, end+0.1, step)):
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
for num1, i in enumerate(D):
    for num2, j in enumerate(i):
        # j为截断阈值，需要进行调整
        if j < 2:
            D[num1][num2] = np.inf

# 将D矩阵元素进行倒数转换
D_pit = np.reciprocal(D)
T_matrix_inv = np.dot(VT.T, D_pit.T.dot(U.T))  # 求解T_matrix的逆矩阵

# 求解光谱矢量
Reconstruction_Spectral_Vector = np.squeeze(T_matrix_inv.dot(Intensity_Vector), axis=1)
# 将光谱矢量中不符合实际物理意义的值抛除
for num, i in enumerate(Reconstruction_Spectral_Vector):
    if i < 0:
        Reconstruction_Spectral_Vector[num] = 0

print('Finish reconstruction . . . ')
print('# # # # # # # # # #\n')

###################################################################################################

Spectral_plot = np.arange(0, end-start+0.1, step) + start

Probe_Intensity_Vector_bak = Probe_Intensity_Vector

# 找到特征光谱在标定光谱中的对应位置，并将其他位置补齐（填充值为0）
print('序号 | list1 | list2 | difference | miu')
for i in range(len(Probe_Intensity_Vector_bak)):
    for j in range(len(Spectral_plot)):
        diff = Probe_Intensity_Vector_bak[i] - Spectral_plot[j]
        if 0 <= diff < step:
            pass

        else:
            print(j, '|', Probe_Intensity_Vector_bak[i], '|', Spectral_plot[j], '|', diff)
            Probe_Intensity_Vector = np.insert(Probe_Intensity_Vector, j, 0)
# 将特征光谱对应位置设置为1，其余位置设置为0
for num, i in enumerate(Probe_Intensity_Vector):
    if i != 0:
        Probe_Intensity_Vector[num] = 1
# 此处用于计算论文中给出的错误率
diff_vector = Probe_Intensity_Vector - Reconstruction_Spectral_Vector

# 求解重建错误率
miu = np.sqrt(np.power(diff_vector, 2).mean()) / (Probe_Intensity_Vector**2).mean()

print('\nReconstruction error is %.4f' % miu)
print('\nFinish calculating reconstruction error . . . ')
print('# # # # # # # # # #\n')

###################################################################################################

# 绘制点列图
plt.figure(0)

plt.stem(Spectral_plot, Reconstruction_Spectral_Vector, use_line_collection=True,
         linefmt='r--', basefmt='b--')

for num, i in enumerate(Reconstruction_Spectral_Vector):  # 标明光谱向量的值
    i = round(i.tolist(), 2)
    if Reconstruction_Spectral_Vector[num] >= 0.1:
        plt.annotate((Spectral_plot[num], i), xy=(Spectral_plot[num], i),
                     xytext=(Spectral_plot[num], i))
plt.grid(), plt.xlabel('wavelength/nm', fontsize=15), plt.ylabel('Correlation', fontsize=15)
print('Plotting figures . . . \n# # # # # # # # # # \nThe End')

plt.show()
