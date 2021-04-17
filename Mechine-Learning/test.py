import torch
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

class BRDb(nn.Module):  # BN_ReLU_Diconv_block，因为无法高效变换通道，可以考虑后续换为Bottleneck块的形式
    def __init__(self, input_channels, output_channels=None, growth_rate=12, kernel=5, dilation_rate=2):
        super(BRDb, self).__init__()
        self.input_channels = input_channels
        self.growth_rate = growth_rate
        self.kernel = kernel
        self.dilation_rate = dilation_rate
        self.output_channels = output_channels
        if not self.output_channels:
            self.output_channels = self.input_channels + self.growth_rate  # 视实际情况更改系数

        self.brdb = nn.Sequential(
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
            # 按照论文上讲，如果输出维度减半，kernel_size=5, 那么padding=k-1, stride=2
            # 但是，这个地方卷积后要保持图片尺寸不变，后边Average Pool的时候尺寸减半，因此，padding=4, stride=1
            nn.Conv2d(self.input_channels, self.output_channels, kernel_size=5, padding=4, dilation=2, bias=False)
            # 视情况将bias设置为False或者重新调整层的顺序，加上一个BN层
        )

    def channels(self):
        return self.output_channels

    def forward(self, x):
        # print('BRDb forward done!')
        return torch.cat([x, self.brdb(x)], 1)


class Dense_Block(nn.Module):
    def __init__(self, input_channels, output_channel=None, block=BRDb, growth_rate=12, kernel=5, dilation_rate=2, nblocks=3):
        super(Dense_Block, self).__init__()
        self.block = block
        self.input_channels = input_channels
        self.output_channels = output_channel
        self.growth_rate = growth_rate
        self.kernel = kernel
        self.dilation_rate = dilation_rate
        self.nblocks = nblocks

        self.feature = nn.Sequential()
        for i in range(nblocks):
            self.feature.add_module('brdb_layer_{}'.format(i), self.block(self.input_channels, self.output_channels, self.kernel, self.dilation_rate))
            self.input_channels = sum([self.feature[t].channels() for t in range(i+1)]) + input_channels
        # 如果觉得通道控制效果不好，在此预留出一个变换通道的1X1卷积层
        # self.input_channels = sum([self.feature[i].channels() for i in range(nblocks)])

        if not self.output_channels:
            self.conv = nn.Conv2d(self.input_channels, input_channels + 3 * self.growth_rate, 1, bias=False)
        else:
            self.conv = nn.Conv2d(self.input_channels, self.output_channels, 1, bias=False)
    def forward(self, x):
        x = self.feature(x)
        # print('Dense_block forward done!')
        return self.conv(x)


# 其output_channels需要根据论文中来定
class Downsampling_block(nn.Module):
    def __init__(self, input_channels, output_channels, downsample_rate=0.05):
        super(Downsampling_block, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.drop_rate = downsample_rate

        self.downsample_block = nn.Sequential(
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_channels, self.output_channels, kernel_size=1, bias=False),
            nn.Dropout2d(p=self.drop_rate, inplace=False),
            # 使用kernel_size=3会比kernel_size=2有更大的计算量，但是kernel为偶数并不是一个好现象
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        # print('Downsampling_block forward done!')
        return self.downsample_block(x)


# 上采样层被封装在torch.nn的vision layer中
# 共包含了四种：①PixelShuffle  ②Upsample   ③UpsampleNearest2d  ④UpsampleBilinear2d
# 了解这四种上采样的区别
class Upsampling_block(nn.Module):
    def __init__(self, input_channels, output_channels=36, upsample_rate=2, growth_rate=12):
        super(Upsampling_block, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.upscale_factor = upsample_rate
        self.growth_rate = growth_rate
        # 此处的通道数论文中未提到
        # inner_channels = self.input_channels + self.growth_rate
        inner_channels = self.input_channels // 4

        self.upsample_block = nn.Sequential(
            nn.Conv2d(self.input_channels, self.input_channels // 4 * 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=self.upscale_factor),
            nn.Conv2d(inner_channels, self.output_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print('upsampling_block forward done!')
        return self.upsample_block(x)


class IDiffNet(nn.Module):
    def __init__(self, i_chs, d_o_chs=(26, 31, 33, 34, 35, 35, 36), g_r=12, d_r=2, n_downsample=6, n_upsample=6):
        super(IDiffNet, self).__init__()

        self.input_channels = i_chs
        self.output_channels = d_o_chs  # 要求是一个包含七个正整数元素的列表
        self.growth_rate = g_r
        self.dilation_rate = d_r

        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=5, padding=4, dilation=2, bias=False)
        self.pool = nn.AvgPool2d(2)
        self.input_channels = 16

        self.down_sample = nn.Sequential()
        for index in range(n_downsample):
            self.down_sample.add_module('dense_block_{}'.format(index), Dense_Block(self.input_channels))
            self.input_channels += 3 * self.growth_rate
            self.down_sample.add_module('downsample_block_{}'.format(index), Downsampling_block(input_channels=self.input_channels, output_channels=self.output_channels[index]))
            self.input_channels = self.output_channels[index]

        # self.input_channels = 35
        self.dense_block = Dense_Block(self.input_channels, 36)


        self.up_sample = nn.Sequential()
        for index in range(n_upsample):
            self.input_channels = self.output_channels[-1] + self.output_channels[5-index]
            self.up_sample.add_module('upsample_block_{}'.format(index), Upsampling_block(input_channels=self.input_channels))
            self.up_sample.add_module('dense_block_{}'.format(index+7), Dense_Block(self.output_channels[-1], self.output_channels[-1]))

        self.add_upsample_block = Upsampling_block(input_channels=self.output_channels[-1])
        self.conv2 = nn.Conv2d(self.output_channels[-1], 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)


        x1 = self.down_sample[0](x)
        x2 = self.down_sample[1](x1)
        x3 = self.down_sample[2](x2)
        x4 = self.down_sample[3](x3)
        x5 = self.down_sample[4](x4)
        x6 = self.down_sample[5](x5)
        x7 = self.down_sample[6](x6)
        x8 = self.down_sample[7](x7)
        x9 = self.down_sample[8](x8)
        x10 = self.down_sample[9](x9)
        x11 = self.down_sample[10](x10)
        x12 = self.down_sample[11](x11)

        x = self.dense_block(x12)

        x = torch.cat([x, x12], 1)
        x = self.up_sample[0](x)
        x = self.up_sample[1](x)
        x = torch.cat([x, x10], 1)
        x = self.up_sample[2](x)
        x = self.up_sample[3](x)
        x = torch.cat([x, x8], 1)
        x = self.up_sample[4](x)
        x = self.up_sample[5](x)
        x = torch.cat([x, x6], 1)
        x = self.up_sample[6](x)
        x = self.up_sample[7](x)
        x = torch.cat([x, x4], 1)
        x = self.up_sample[8](x)
        x = self.up_sample[9](x)
        x = torch.cat([x, x2], 1)
        x = self.up_sample[10](x)
        x = self.up_sample[11](x)

        x = self.add_upsample_block(x)
        x = self.conv2(x)
        output = self.relu(x)
        # print('IDiffNet forward done!')
        return output


def idiffnet(i_chs=1):
    return IDiffNet(i_chs, d_o_chs=(26, 31, 33, 34, 35, 35, 36), g_r=12, d_r=2, n_downsample=6, n_upsample=6)


if __name__ == '__main__':
    i = 6
    net = torch.load('./net.pkl', torch.device('cpu'))
    image = Image.open('./image/TestImage_' + str(i) + '.png')
    label = Image.open('./label/TestImage_' + str(i) + '.png')

    transformer = transforms.ToTensor()
    input = torch.unsqueeze(transformer(image), dim=0)

    net.eval()
    output = net(input)

    output = torch.squeeze(output, dim=0).detach().numpy()
    plt.subplot(131)
    plt.imshow(image, cmap=plt.cm.BuPu_r)
    plt.title('experiment')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132)
    plt.imshow(output[0], cmap=plt.cm.BuPu_r)
    plt.title('network-output')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133)
    plt.imshow(label, cmap=plt.cm.BuPu_r)
    plt.title('label')
    plt.xticks([]), plt.yticks([])


    # n = 8
    # net = torch.load('./net.pkl', torch.device('cpu'))
    # images, labels = [], []
    # transformer = transforms.ToTensor()
    # for i in range(n):
    #     image = Image.open('./image/TestImage_' + str(i+1) + '.png')
    #     image = transformer(image)
    #     if i == 0:
    #         images = image
    #         continue
    #     images = torch.cat([images, image])
    #     label = Image.open('./label/TestImage_' + str(i+1) + '.png')
    #     labels.append(label)
    #
    # images = torch.unsqueeze(images, dim=1)
    # outputs = net(images)
    #
    # images = torch.squeeze(images, dim=1).detach().numpy()
    # outputs = torch.squeeze(outputs, dim=1).detach().numpy()
    #
    # _, figs1 = plt.subplots(1, len(images))
    # plt.title('experiments')
    # for f, img in zip(figs1, images):
    #     f.imshow(img)
    #     f.axes.get_xaxis().set_visible(False)
    #     f.axes.get_yaxis().set_visible(False)
    #
    # _, figs2 = plt.subplots(1, len(outputs))
    # plt.title('outputs')
    # for f, img in zip(figs2, outputs):
    #     f.imshow(img)
    #     f.axes.get_xaxis().set_visible(False)
    #     f.axes.get_yaxis().set_visible(False)
    #
    # _, figs3 = plt.subplots(1, len(labels))
    # plt.title('labels')
    # for f, img in zip(figs3, labels):
    #     f.imshow(img)
    #     f.axes.get_xaxis().set_visible(False)
    #     f.axes.get_yaxis().set_visible(False)

    plt.show()