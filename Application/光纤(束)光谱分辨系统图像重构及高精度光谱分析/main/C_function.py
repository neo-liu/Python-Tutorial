# author: Neo Liu
# data: 2020.4.9
# function: 处理光强分布，获取直接的Probe光谱
from tools import *

# 定义超参数，用来控制整个程序的参数，可以根据不同情况灵活控制
name1 = './data format.txt'
name2 = './output_755.0.fld'  # 此为标定谱线的光强分布数据之一，用于读取数据维度使用
start, end, step = 755.0, 780.0+1e-5, 0.2  # 此为波长参数
delta_lambda_start, delta_lambda_end, delta_lambda_step = 0.00, 0.05, 0.01

# 波长（横）坐标， 以及平滑C时使用的波长（横）坐标
delta_lambda = np.arange(0, delta_lambda_end - delta_lambda_start, delta_lambda_step)
Lambda_Smooth = np.linspace(min(delta_lambda), max(delta_lambda), 1000)

# 读取光强分布的数据格式
boundary = read_format(name1)
XY_num_diff = int(boundary[1] - boundary[0])

# 获取光强分布数据的维度
shape = list(get_shape(name2))

# read all data of Intensity distribution
for num, delta in enumerate(delta_lambda):
    print('Loading data while Δ = ' + str(delta) + '\n # # # # # # # # # # \n')
    for num1, i in enumerate(np.arange(start + delta, end + delta, step)):
        data = read_file('./output_' + str(round(i, 2)) + '.fld', data, shape, multi=True, num=num1)
    for j in data:
        j = j[:, int(XY_num_diff/2):int(boundary[1] - XY_num_diff/2)]
        Data.append(j[1:-1, 1:-1].reshape(1, -1)[0])

    Data_set.append(np.expand_dims(Data, axis=0))
    Data.clear()
    data = np.array([])
Data_set = np.squeeze(Data_set, axis=1)

# calculate correlation
for i in Data_set:
    up = np.mean(np.multiply(Data_set[0], i), axis=0)
    down = np.multiply(np.mean(Data_set[0], axis=0), np.mean(i, axis=0))
    Correlation.append(np.mean((up / down), axis=0) - 1)
for i in np.arange(len(Correlation)):
    Correlation[i] = Correlation[i] / max(Correlation)
# smooth operation(cubic)
Correlation_Smooth = interp1d(delta_lambda, np.array(Correlation), kind='cubic')  # quadratic

# svg display
display.set_matplotlib_formats('svg')

# setup figure and plot
plt.figure(figsize=(7, 6))
plt.plot(Lambda_Smooth, Correlation_Smooth(Lambda_Smooth), linewidth=2, color='green', linestyle='-')
print(min(Correlation_Smooth(Lambda_Smooth)))
# plot scatter and reference
for num, i in enumerate(Correlation_Smooth(Lambda_Smooth)):
    if i < (Correlation[0] / 2):
        plt.scatter(Lambda_Smooth[num], (Correlation[0] / 2),
                    color='red', marker='o', linewidths=3)
        plt.scatter(Lambda_Smooth[2*num], Correlation_Smooth(Lambda_Smooth)[2*num],
                    color='red', marker='o', linewidths=3)
        # represent resolution
        resolution = Lambda_Smooth[2*num] - min(Lambda_Smooth)
        plt.axvline(Lambda_Smooth[2*num], 0, 0.15, linestyle='--')
        plt.axhline(Correlation_Smooth(Lambda_Smooth)[2*num], 0, 0.06, linestyle='--')

        break

# format correlation
for num, i in enumerate(Correlation):
    Correlation[num] = float(str(Correlation[num])[:4])

for i in range(len(delta_lambda)):
    plt.annotate(Correlation[i], xy=(delta_lambda[i], Correlation[i]), xytext=(delta_lambda[i] * 1.0001,
                                                                               Correlation[i]*1))

plt.grid(), plt.title('Resolution : ' + str(resolution)[0:4] + 'nm', fontsize=25)
plt.xlabel('wavelength(nm)', fontsize=15), plt.ylabel('Spectral Correlation C(d lambda)', fontsize=15)

plt.show()
