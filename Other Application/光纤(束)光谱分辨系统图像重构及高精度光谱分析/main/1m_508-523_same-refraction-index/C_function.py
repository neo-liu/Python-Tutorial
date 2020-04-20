'''
author: Neo Liu
data: 2020.4.9
function: 处理光强分布，获取直接的Probe光谱
'''
from tools import *

# 定义超参数，用来控制整个程序的参数，可以根据不同情况灵活控制
name1 = './data format.txt'
name2 = './output_508.0.fld'  # 此为标定谱线的光强分布数据之一，用于读取数据维度使用
start, end, step = 521.2, 523.0, 0.2  # 此为波长参数

# 波长（横）坐标， 以及平滑C时使用的波长（横）坐标
Lambda = np.arange(0, end-start, step) + start
Lambda_Smooth = np.linspace(Lambda.min(), Lambda.max(), 500)

# 读取光强分布的数据格式
boundary = read_format(name1)
XY_num_diff = int(boundary[1] - boundary[0])

# 获取光强分布数据的维度
shape = list(get_shape(name2))

# read all data of Intensity distribution
for num, i in enumerate(np.arange(start, end, step)):
    Data = read_file('./output_' + str(round(i, 4)) + '.fld', shape, multi=True, num=num)

for num, i in enumerate(Data):
    i = i[:, int(XY_num_diff/2):int(boundary[1] - XY_num_diff/2)].reshape(1, -1)
    Data[num] = i[0]

# base wavelength
data = Data[0]

# A_P_1 represent <I(λ，x)>
# A_P_1 = data.mean()
A_P_1 = data / Lambda[0]

for num, i in enumerate(Data):
    # P_A represent <I(λ，x)I(λ+Δλ，x)>
    # P_A = (data * i).mean()
    P_A = np.dot(data, i) / Lambda[num]

    # A_P_2 represent <I(λ+Δλ，x)>
    # A_P_2 = i.mean()
    A_P_2 = i / Lambda[num]
    # print(P_A, A_P_1, A_P_2)
    # correlation equals <I(λ，x)I(λ+Δλ，x)>/[<I(λ，x)><I(λ+Δλ，x)>] - 1
    Correlation.append((P_A / A_P_1 / A_P_2 - 1).mean())
    # Correlation.append(np.corrcoef(data, i).tolist()[0][1])
    print(P_A, A_P_1.shape, A_P_2.shape, Correlation[num])

# smooth operation(cubic)
Correlation_Smooth = interp1d(Lambda, Correlation, kind='cubic')

# svg display
display.set_matplotlib_formats('svg')

# setup figure and plot
plt.figure(figsize=(7, 6))
plt.plot(Lambda_Smooth, Correlation_Smooth(Lambda_Smooth), linewidth=2, color='green', linestyle='-')

# plot scatter and reference
for num, i in enumerate(Correlation_Smooth(Lambda_Smooth)):
    if i < (Correlation[0] / 2):
        plt.scatter(Lambda_Smooth[num], (Correlation[0] / 2),
                    color='red', marker='o', linewidths=3)
        plt.scatter(Lambda_Smooth[2*num], Correlation_Smooth(Lambda_Smooth)[2*num],
                    color='red', marker='o', linewidths=3)
        # represent resolution
        resolution = Lambda_Smooth[2*num] - Lambda_Smooth.min()
        plt.axvline(Lambda_Smooth[2*num], 0, 0.15, linestyle='--')
        plt.axhline(Correlation_Smooth(Lambda_Smooth)[2*num], 0, 0.06, linestyle='--')

        break

# format correlation
for num, i in enumerate(Correlation):
    Correlation[num] = float(str(Correlation[num])[:4])

for i in range(len(Lambda)):
    plt.annotate(Correlation[i], xy=(Lambda[i], Correlation[i]),
                                xytext=(Lambda[i] * 1.0001, Correlation[i]*1))

plt.grid(), plt.title('Resolution : ' + str(resolution)[0:4] + 'nm', fontsize=25)
plt.xlabel('wavelength(nm)', fontsize=15), plt.ylabel('Spectral Correlation C(d lambda)',
                                                                        fontsize=15)

plt.show()