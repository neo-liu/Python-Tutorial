'''
author：刘有为
notes:代码中带有数字‘1’的为实验相关变量；带有数字‘2’的为理论相关变量
'''
import cv2
import numpy as np 
import math
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

##########################################################################
# 形态学运算（膨胀）时使用的滤波核
kernel = np.ones((2,2),np.uint8)
# 定义存放角度（弧度）的列表
Beta1 = [0] 
Beta2 = [ 330, 340, 350, 0  , 10 , 20 , 30 , 40 , 50 , 60 , 70 , 80 , 90 , 100, 110,
          120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230,
          240, 250, 260, 270, 280, 290, 300, 310, 320]
# 定义存放rho的列表 
rho1 = list()
rho2 = list()
Centroids = list()


# 计算I0、alpha、Ao、Ae
I0 = 0.667 * 2
alpha = math.pi / 6
Ao = math.sqrt(I0) * math.sin(alpha)
Ae = math.sqrt(I0) * math.cos(alpha)
#定义计算rho的函数，可直接返回rho
def Rho1(I):
    return math.sqrt((Ao**2 * Ae**2) / (Ao**2 + Ae**2 - I))

def Rho2(beta):
    I = Ae**2 * (math.cos(beta))**2 + Ao**2 * (math.sin(beta))**2
    return Rho1(I)

####################################################################
#第一部分结束
####################################################################

# 将三十六张图片依次处理 
for i in range(1, 37):
    # 导入图像
    image = cv2.imread('./'+str(i)+'.jpg',  255)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 灰度转化

    # 提取暗楔形区域
    ret, mask = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img2_fg = cv2.bitwise_and(img,img,mask = mask_inv)
    dilation = cv2.dilate(img2_fg,kernel,iterations = 1)
    # 读出成功输出
    print('Downloading complished！ ==>' + str(i))
    #创建两个存放像素的列表
    A = list()      
    B = list()
    # 列表转换
    List = np.array(dilation)
    List = list(List)
    for i in List:
        i = list(i)
        A.append(i)
    # 对像素值进行判断
    for n, i in enumerate(A):
        for m, j in enumerate(i):
            if A[n][m] > 40:
                B.append((n, m))

    # K值聚类
    estimator = KMeans(n_clusters=2)
    res = estimator.fit_predict(B)
    lable_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    inertia = estimator.inertia_
    #print((centroids[0][0], centroids[0][1]), (centroids[1][0], centroids[1][1]))
    # 计算聚类两点的斜率对应的角度（弧度制）
    anglea = abs(-math.pi/2 + math.atan2(centroids[0][0]-centroids[1][0], centroids[0][1]-centroids[1][1]))
    angleb = abs(-math.pi/2 + math.atan2(centroids[1][0]-centroids[0][0], centroids[1][1]-centroids[0][1]))
    if abs(anglea*180/math.pi)>180:
        anglea = anglea - math.pi
    elif Beta1[-1]*180/math.pi>90 and (anglea*180/math.pi)<90:
        anglea = angleb
    Beta1.append(anglea)
    angle_jiaodu = abs(anglea*180/math.pi)
    print('弧度a为===>', anglea)
    print('弧度b为===>', angleb)    
    print('角度为===>', angle_jiaodu)
    # 将成对的散点画出，且每次画出的时候输出现在画出散点的个数
    for n,i in enumerate(range(0, len(B), 80)):
        #print('for ==>',n+1,'calculations')
        # 绘制类型一的散点
        if int(lable_pred[i])==0:
            plt.scatter(B[i][1],B[i][0],color='green')     
        # 绘制类型二的散点 
        elif int(lable_pred[i])==1:
            plt.scatter(B[i][1],B[i][0],color='blue')
    # 画出聚类重心
    #plt.scatter(centroids[0][1],centroids[0][0],color='red')
    #plt.scatter(centroids[1][1],centroids[1][0],color='black')
    # 连接聚类重心
    #plt.plot((centroids[0][1],centroids[1][1]), (centroids[0][0],centroids[1][0]), lw=5, color='green')
    # 图片参数
    #plt.xlabel('longitude')
    #plt.ylabel('latitude')
   # plt.title('Angle is '+str(angle_jiaodu)+'image'+str(i))
    #plt.show()
    #plt.savefig('./n'+str(i)+'.png')
    # 原图参数，不要也罢，不是很影响程序功能
    #cv2.line(image, (int(centroids[1][0]), int(centroids[1][1])),(int(centroids[0][0]), int(centroids[0][1])), (255, 0, 0), 2) 
    #cv2.namedWindow('IMG', cv2.WINDOW_NORMAL)
    #cv2.imshow('IMG',img2_fg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# 测量得出的光强
# 珉组
#I = [0.635, 0.675, 0.680, 0.647, 0.575, 0.483, 0.369, 0.258, 0.152, 
#     0.079, 0.032, 0.028, 0.061, 0.135, 0.225, 0.345, 0.451, 0.551,
#     0.635, 0.678, 0.681, 0.652, 0.587, 0.486, 0.381, 0.267, 0.163,
#     0.084, 0.036, 0.027, 0.061, 0.128, 0.231, 0.343, 0.445, 0.549]
Beta1.pop(0)
for i in range(19, 36):
    Beta1[i] = Beta1[i]+math.pi
I = [0.667,0.538,0.438,0.330,0.216,0.126,0.062,0.032,0.040 ,
     0.086,0.168,0.266,0.373,0.483,0.572,0.641,0.669,0.664,
     0.620,0.545,0.446,0.334,0.228,0.133,0.065,0.033,0.040,
     0.086,0.155,0.257,0.368,0.467,0.567, 0.325, 0.434, 0.533]
# 计算出每个光强测量值对应的rho值
for i in I:
    rho1.append(Rho1(i))
for i in Beta2:
    rho2.append(Rho2(i))
# 角度列表
theta1 = Beta1
theta2 = Beta2 
# 散点的大小（面积）
area = 60
# 散点的颜色（随机）
colors = 2 * np.random.rand(36)
# 绘制坐标类型
ax = plt.subplot(121, projection='polar')
plt.title('plot expriment image')
bx = plt.subplot(122, projection='polar')
plt.title('ideal data image')
# 绘制坐标参数
c = ax.scatter(theta1, rho1, c=colors, s=area, cmap='hsv', alpha=0.75)
d = bx.scatter(theta2, rho2, c=colors, s=area, cmap='hsv', alpha=0.75)
plt.show()
