from collections import OrderedDict

import torch
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d

# Detector本身，不同层级特征输入，进行融合后再进行预测
class SegDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],  # Res不同层级，输入的通道数
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels // 4, 3, padding=1, bias=bias)
        # 利用组合特征，得到置信度的网络（这里实际上不是二值化）
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    # 阈值网络
    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
    # features 分别为 (batch_size, 64, 160, 160), (batch_size, 128, 80, 80), (batch_size, 256, 40, 40), (batch_size, 512, 20, 20)
    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features  # ResNet传过来的特征，channel分别为之前设定好的64,128,256,512，in网络只有一个conv，相当于调整channel都变成256
        in5 = self.in5(c5)  # in5 (batch_size, 256, 20, 20)
        in4 = self.in4(c4)  # in4 (batch_size, 256, 40, 40)
        in3 = self.in3(c3)  # in3 (batch_size, 256, 80, 80)
        in2 = self.in2(c2)  # in2 (batch_size, 256, 160, 160)
        # up网络就是UpSample，即将下层的特征放大，加到上一层里
        out4 = self.up5(in5) + in4  # 1/16  out4 (batch_size, 256, 40, 40)
        out3 = self.up4(out4) + in3  # 1/8  out3 (batch_size, 256, 80, 80)
        out2 = self.up3(out3) + in2  # 1/4  out2 (batch_size, 256, 160, 160)
        # out网络先通过卷积将channel变为1/4（256-->64），然后通过不同的UpSample倍率，使得所有的特征缩放到统一大小
        p5 = self.out5(in5)  # p5 (batch_size, 64, 160, 160)
        p4 = self.out4(out4)  # p4 (batch_size, 64, 160, 160)
        p3 = self.out3(out3)  # p3 (batch_size, 64, 160, 160)
        p2 = self.out2(out2)  # p2 (batch_size, 64, 160, 160)
        # fuse (batch_size, 256, 160, 160)，在第一维上拼接出来的融合了不同层级的特征
        fuse = torch.cat((p5, p4, p3, p2), 1)  # 拼接不同层级的特征，得到融合特征
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)  # binary (batch_size, 1, 640, 640)，这个实际上不是二值化结果，是float型的，可以认为是置信度，参考原始注释
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary  # 如果是测试，只需要返回binary就可以输出了
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                    (fuse, nn.functional.interpolate(
                        binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)  # thresh (batch_size, 1, 640, 640) 实际上thresh网络结构和binarize是一模一样的，只是一个输出置信度，一个输出阈值
            thresh_binary = self.step_function(binary, thresh)  # thresh_binary (batch_size, 1, 640, 640) 调用论文里的那个核心公式，进行可微分二值化
            result.update(thresh=thresh, thresh_binary=thresh_binary)  # 训练的时候，则返回所有的计算值
        return result
    # 核心公式
    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
