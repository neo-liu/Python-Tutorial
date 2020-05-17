# author: Neo Liu
# data: 2020.4.9
# function: 处理光强分布，获取直接的Probe光谱
from tools import *  # 导入读取数据、计算需要使用到的函数包

###################################################################################################

# 定义参数，用来控制整个程序的参数，可以根据不同情况灵活控制
name1 = './data format.txt'  # 此为数据格式文件
name2 = './output_755.0.fld'  # 此为标定谱线的光强分布数据之一，用于读取数据维度使用

# 此为波长覆盖参数（对应与数据表的横向维度），波长范围λ从755.0至780.0nm，波长间隔为0.2nm，共126组
start, end, step = 755.0, 780.0+1e-4, 0.2  # 用于还原光谱

# 此为波长细微变化参数（对应于数据表的纵向维度），Δλ从0至0.05nm变化，间隔为0.01nm, 共6组
delta_lambda_start, delta_lambda_end, delta_lambda_step = 0.00, 0.08, 0.01

###################################################################################################
# 读取数据

# 波长（横）坐标， 以及平滑C时使用的波长（横）坐标，作为光谱相关度曲线的横坐标使用
delta_lambda = np.arange(0, delta_lambda_end - delta_lambda_start, delta_lambda_step)
Lambda_Smooth = np.linspace(min(delta_lambda), max(delta_lambda), 1000)

# 读取光强分布的数据格式，用于后边散斑图的整形
boundary = read_format(name1)
XY_num_diff = int(boundary[1] - boundary[0])

# 获取光强分布数据的维度，用于后边散斑图的整形和验证读取数据的正确性
shape = list(get_shape(name2))

# 读取所有数据（具体细节我认为不是很重要，而且数据结构上稍稍复杂）
for num, delta in enumerate(delta_lambda):
    print('Loading data while Δ = ' + str(delta) + '\n # # # # # # # # # # \n')
    # 每个循环中都读取一个Δλ取值的数据
    for num1, i in enumerate(np.arange(start + delta, end + delta, step)):
        data = read_file('./output_' + str(round(i, 4)) + '.fld', data, shape, multi=True, num=num1)
    # 对数据做一些整形，展开的处理
    for j in data:
        j = j[:, int(XY_num_diff/2):int(boundary[1] - XY_num_diff/2)]  # 将数据整形成方形
        Data.append(j[:, :].reshape(1, -1)[0])  # 将数据展开成一个一维向量
    Data_set.append(np.expand_dims(Data, axis=0))
    Data.clear()
    data = np.array([])

# Data_set为读取数据的最后结果，数据维度为[6， 126， 75076]
# 数据说明：为了便于计算，在读取数据后，将二维的散斑数据展开成了一维形式
# ’6‘维度：由代码第16行控制；’126‘维度：由第13行控制；‘75076’维度：由数据格式控制，代表每个光强分布中包含75076组数据
Data_set = np.squeeze(Data_set, axis=1)

###################################################################################################
# 计算光谱相关度

# 光谱相关度计算 C（Δλ，x） = <I(λ，x)I(λ+Δλ，x)> / [<I(λ，x)><I(λ+Δλ，x)>] - 1
# 其中<>代表对光强在λ上取平均，<I(λ，x)> = [I(λ1，x)+I(λ2，x)+ ... I(λn，x)] / n ；   <I(λ+Δλ，x)>同理
# <I(λ，x)I(λ+Δλ，x)> = [I(λ1，x)I(λ1+Δλ，x)+I(λ2，x)I(λ2+Δλ，x)+ ... +I(λn，x)I(λn+Δλ，x)] / n
# 另外对λ取平局后，需要对整体在x上取平均
for i in Data_set:
    # Data_set[0]为Δλ=0时的光强分布，共包含126组散斑数据，维度形式[126, 75076]
    # 在循环中i分别表示Δλ=0，0.01，0.02，0.03，0.04，0.05时的光强分布，每个i均包含126组数据，维度形式[126, 75076]
    # 在求均值令axis=0时，可得到光强分布在λ上的平均值
    # 注：np.multiply 求解的是两个矩阵元素的乘积，而不是矩阵乘法。矩阵乘法表示为np.dot
    up = np.mean(np.multiply(Data_set[0], i), axis=0)  # 计算<I(λ，x)I(λ+Δλ，x)>
    down = np.multiply(np.mean(Data_set[0], axis=0), np.mean(i, axis=0))  # 计算I(λ，x)><I(λ+Δλ，x)>
    Correlation.append(np.mean((up / down), axis=0) - 1)  # 最后对x维度取均值，计算光谱相关度

# 使用插值的方式，将光谱相关度曲线拟合成为二阶、三阶或者更高阶曲线
Correlation_Smooth = interp1d(delta_lambda, np.array(Correlation), kind='cubic')  # quadratic

###################################################################################################
# 绘制光谱相关度曲线

# 使用矢量图方式画出拟合曲线
display.set_matplotlib_formats('svg')

# 各种图表参数
plt.figure(figsize=(7, 6))
plt.plot(Lambda_Smooth, Correlation_Smooth(Lambda_Smooth), linewidth=2, color='green', linestyle='-')
print(min(Correlation_Smooth(Lambda_Smooth)))

# plot scatter and reference
for num, i in enumerate(Correlation_Smooth(Lambda_Smooth)):
    if i < (Correlation[0] / 2):
        # 绘制初值下降一半时的散点
        plt.scatter(Lambda_Smooth[num], (Correlation[0] / 2),
                    color='red', marker='o', linewidths=3)
        # 绘制光谱相关宽度的位置，如果找不到则会报错
        # 在运行的时候应该先注释掉，先判断、后使用
        # plt.scatter(Lambda_Smooth[2*num], Correlation_Smooth(Lambda_Smooth)[2*num],
        #             color='red', marker='o', linewidths=3)
        # represent resolution
        resolution = 2*Lambda_Smooth[num] - min(Lambda_Smooth)
        # plt.axvline(Lambda_Smooth[2*num], 0, 0.15, linestyle='--')
        # plt.axhline(Correlation_Smooth(Lambda_Smooth)[2*num], 0, 0.06, linestyle='--')

        break

# 在图中标记光谱相关度值
for num, i in enumerate(Correlation):
    Correlation[num] = float(str(Correlation[num])[:4])
for i in range(len(delta_lambda)):
    plt.annotate(Correlation[i], xy=(delta_lambda[i], Correlation[i]), xytext=(delta_lambda[i] * 1.0001,
                                                                               Correlation[i]*1))

plt.grid(), plt.title('Resolution : ' + str(resolution)[0:4] + 'nm', fontsize=25)
plt.xlabel('wavelength(nm)', fontsize=15), plt.ylabel('Spectral Correlation C(d lambda)', fontsize=15)

plt.show()
