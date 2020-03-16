import numpy as np
import matplotlib.pyplot as plt

# 定义存储列表
T_matrix = list()
data = list()
boundary = list()
Intensity_Vector = list()
Spectral_Vector = list()

with open('./data format.txt', 'r') as f:  # 读取格式文件
    for line in f:
        line = [float(i) for i in line.split(' ')]
        boundary.append(line)
boundary = np.array(boundary)
XY_num_diff = int(boundary[1][0] - boundary[0][0])

with open('./test.txt', 'r') as f:
    for line in f:
        line = [float(i) for i in line.split(' ')[1: -1]]
        Intensity_Vector.append(line)
Intensity_Vector = np.array(Intensity_Vector[1:])

Intensity_Vector = Intensity_Vector[:,
                   int(XY_num_diff/2):int(boundary[1][0] - XY_num_diff/2)].reshape(-1, 1)

for i in range(0, 31):  # 读取所有数据
    with open('./'+str(i)+'.txt', 'r') as f:
        for line in f:
            line = [float(i) for i in line.split(' ')[1: -1]]
            data.append(line)

    T_matrix.append(np.array(data)[1:])
    T_matrix[i] = T_matrix[i][:, int(XY_num_diff/2):int(boundary[1][0] - XY_num_diff/2)]
    data.clear()

for i in range(len(T_matrix)):
    T_matrix[i] = T_matrix[i].reshape(1, -1)   # / 255.0
T_matrix = np.array(np.squeeze(T_matrix, axis=1)).T  # 格式数据

# 将T_matrix中过小的元素进行截断
for num1, i in enumerate(T_matrix):
    for num2, j in enumerate(i):
        if j <= 0.002:
            T_matrix[num1][num2] = 0

# 求解Data的截断伪逆矩阵
T_matrix_inv = np.linalg.pinv(T_matrix)
print(T_matrix_inv)

# 求解光谱矢量
Spectral_Vector = np.array(T_matrix_inv.dot(Intensity_Vector))

# Spectral1 = np.arange(0, 1.1, 0.1)
# Spectral2 = np.arange(10, 110, 10)
# Spectral = np.append(Spectral1, Spectral2, axis=0) + 1550
Spectral = np.arange(0, 6.2, 0.2) + 765
# 绘制点列图
plt.stem(Spectral, Spectral_Vector, use_line_collection=True, linefmt='r--', basefmt='b--')
# plt.stem([1560.9], [1], use_line_collection=True, linefmt='y-', basefmt='b--', )
for num, i in enumerate(Spectral_Vector):  # 标明光谱向量的值
    i = round(i.tolist()[0], 2)
    if Spectral_Vector[num] >= 0.05:
        plt.annotate(i, xy=(Spectral[num], i)
                     , xytext=(Spectral[num]*1.02, i*1.02))
plt.grid(); plt.xlabel('wavelength/nm', fontsize=15); plt.ylabel('Correlation', fontsize=15)
plt.show()