import math

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


# https://blog.csdn.net/qq_44864833/article/details/128976584
class SKConv_new(nn.Module):
    def __init__(self, features, WH, G, r, M=2, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.  输入特征的维度
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.            卷积组个数？？？
            r: the radio for compute d, the length of z.  用于计算D的比率，z的长度
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv_new, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):  # 初始化两个大小不同的感受野
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))  # 第一个小感受野，第二个为大感受野
        self.gap = nn.AvgPool2d(int(WH / stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, local, globel):  # 1 64 128 128
        for i, conv in enumerate(self.convs):
            if i == 0:
                fea = conv(local).unsqueeze_(dim=1)  # unsqueeze_增加一个维度  # 1 1 64 128 128
                feas = fea
            else:
                fea = conv(globel).unsqueeze_(dim=1)  # 1 1 64 128 128
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)  # 1 64 128 128
        fea_s = self.gap(fea_U)  # squeeze_减少一个模块
        fea_s = fea_s.squeeze(2).squeeze(2)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


'''-------------一、SE模块-----------------------------'''


# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
        self.conv1_1 = nn.Conv2d(inchannel,1,kernel_size=1)

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return self.conv1_1(x * y.expand_as(x))


'''-------------二、ECA模块-----------------------------'''


class eca_block(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel, b=1, gama=2):
        # 继承父类初始化
        super(eca_block, self).__init__()

        # 根据输入通道数自适应调整卷积核大小
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        # 如果卷积核大小是奇数，就使用它
        if kernel_size % 2:
            kernel_size = kernel_size
        # 如果卷积核大小是偶数，就把它变成奇数
        else:
            kernel_size = kernel_size + 1

        # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
        padding = kernel_size // 2

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        # 这个1维卷积需要好好了解一下机制，这是改进SENet的重要不同点
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        # sigmoid激活函数，权值归一化
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, h, w = inputs.shape

        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
        x = x.view([b, 1, c])  # 这是为了给一维卷积
        # 1D卷积 [b,1,c]==>[b,1,c]
        x = self.conv(x)
        # 权值归一化
        x = self.sigmoid(x)
        # 维度调整 [b,1,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
        outputs = x * inputs
        return outputs


'''-------------三、CBAM模块-----------------------------'''


# （1）通道注意力机制
class cbam_block(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(cbam_block, self).__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x

        return outputs


'''-------------四、SKNet模块-----------------------------'''


class SKConv(nn.Module):
    def __init__(self, in_ch, M=3, G=1, r=4, stride=1, L=32) -> None:
        super().__init__()
        """ Constructor
        Args:
        in_ch: input channel dimensionality.
        M: the number of branchs.
        G: num of convolution groups.
        r: the radio for compute d, the length of z.
        stride: stride, default 1.
        L: the minimum dim of the vector z in paper, default 32.
        """
        d = max(int(in_ch / r), L)  # 用来进行线性层的输出通道，当输入数据In_ch很大时，用L就有点丢失数据了。
        self.M = M
        self.in_ch = in_ch
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True)
                )
            )
        # print("D:", d)
        self.fc = nn.Linear(in_ch, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, in_ch))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):  # 第一部分，每个分支的数据进行相加,虽然这里使用的是torch.cat，但是后面又用了unsqueeze和sum进行升维和降维
            fea = conv(x).clone().unsqueeze_(dim=1).clone()  # 这里在1这个地方新增了一个维度  16*1*64*256*256
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas.clone(), fea], dim=1)  # feas.shape  batch*M*in_ch*W*H
        fea_U = torch.sum(feas.clone(), dim=1)  # batch*in_ch*H*W
        fea_s = fea_U.clone().mean(-1).mean(-1)  # Batch*in_ch
        fea_z = self.fc(fea_s)  # batch*in_ch-> batch*d
        for i, fc in enumerate(self.fcs):
            # print(i, fea_z.shape)
            vector = fc(fea_z).clone().unsqueeze_(dim=1)  # batch*d->batch*in_ch->batch*1*in_ch
            # print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors.clone(), vector], dim=1)  # 同样的相加操作 # batch*M*in_ch
        attention_vectors = self.softmax(attention_vectors.clone())  # 对每个分支的数据进行softmax操作
        attention_vectors = attention_vectors.clone().unsqueeze(-1).unsqueeze(-1)  # ->batch*M*in_ch*1*1
        fea_v = (feas * attention_vectors).clone().sum(dim=1)  # ->batch*in_ch*W*H
        return fea_v


'''-------------六、scSE模块-----------------------------'''
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
import torch
import torchvision.transforms as transforms
import math


class sSE(nn.Module):  # 空间(Space)注意力
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        q = self.conv(x)  # b c h w -> b 1 h w
        q = self.norm(q)  # b 1 h w
        return x * q  # 广播机制


class cSE(nn.Module):  # 通道(channel)注意力
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # b c 1 1
        self.relu = nn.ReLU()
        self.Conv_Squeeze = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
        self.Conv_Excitation = nn.Conv2d(in_ch // 2, in_ch, kernel_size=1, bias=False)

    def forward(self, x):
        z = self.avgpool(x)  # b c 1 1
        z = self.Conv_Squeeze(z)  # b c//2 1 1
        z = self.relu(z)
        z = self.Conv_Excitation(z)  # b c 1 1
        z = self.norm(z)
        return x * z.expand_as(x)  # 扩展


class scSE(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.cSE = cSE(in_ch)
        self.sSE = sSE(in_ch)
        self.conv1_1 = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        c_out = self.cSE(x)
        s_out = self.sSE(x)
        return self.conv1_1(c_out + s_out)


'''-------------七、Non-Local Net模块-----------------------------'''


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True) -> None:
        super().__init__()
        """
        in_channels: 输入通道
        inter_channels: 中间数据通道
        dimension: 输入数据的维度
        sub_sample: 是否进行最大池化 一般是True
        bn_layer: 一般是True
        """
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)  # 使用 0 对 参数进行赋初值
            nn.init.constant_(self.W[1].bias, 0)  # 使用 0 对参数进行赋初值
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels,
                             -1)  # b c w*h  这里还经过了maxpool的操作，maxpool:w_out = (w-k_size+2*pad)/k_size + 1
        print(g_x.shape, "self.g后的数据")
        g_x = g_x.permute(0, 2, 1)  # 维度变化 b wh c

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # b c w*h 这里没有经过maxpool操作
        print(theta_x.shape, "self.theta后的数据")
        theta_x = theta_x.permute(0, 2, 1)  # b wh c

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # b c w*h 这里经过了maxpool
        print(phi_x.shape, "self.phi_x后的数据")

        f = torch.matmul(theta_x, phi_x)  # 1024*8 矩阵乘 8*256  =  1024*256

        print(f.shape)

        f_div_C = F.softmax(f, dim=-1)  # 对 最后一维做softmax

        y = torch.matmul(f_div_C, g_x)  # 1024*256  *   256*8   = 1024*8
        print(y.shape, "g_x和y矩阵乘后的结果")
        y = y.permute(0, 2, 1).contiguous()  # 这里的contiguous类似与clone 否则后期对y修改数据，也会对原始数据进行修改  # 得到 batch_size*8*1024
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # *x.size()[2:] 这个花里胡哨的，就是获取x的h 和 w; 再将数据恢复到原始格式
        W_y = self.W(y)  # 这里将b inter_ch w h -> b in_ch w h
        z = W_y + x  # 进行残差连接
        return z


'''-------------八、GCNet模块-----------------------------'''


class GlobalContextBlock(nn.Module):
    """
    inplanes 输入通道数
    ratio 输入通道数和输出通道数的比例
    """

    def __init__(self, inplanes, ratio, pooling_type="att", fusion_types=('channel_mul')) -> None:
        super().__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        # assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None

        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':  # 这里其实就是空间注意力 最后得到一个b c 1 1的权重
            input_x = x
            input_x = input_x.view(batch, channel, height * width)  # -> b c h*w
            input_x = input_x.unsqueeze(1)  # -> b 1 c hw
            context_mask = self.conv_mask(x)  # b 1 h w
            context_mask = context_mask.view(batch, 1, height * width)  # b 1 hw
            context_mask = self.softmax(context_mask)  # b 1 hw
            context_mask = context_mask.unsqueeze(-1)  # b 1 hw 1
            context = torch.matmul(input_x, context_mask)  # b(1 c hw  *  1 hw 1) -> b 1 c 1
            context = context.view(batch, channel, 1, 1)  # b c 1 1
        else:
            context = self.avg_pool(x)  # b c 1 1
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))  # 将权重进行放大缩小
            out = out * channel_mul_term  # 与x进行相乘
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


'''-------------九、ASFF模块-----------------------------'''


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ASFF(nn.Module):  # level只有0、1、2
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=0.5
        1024, 512, 256 -> multiplier=1
        For even smaller, you need change code manually.
        """
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier),
                    int(256 * multiplier)]
        # print(self.dim)

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2)  # size is half
            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)  # size is hale
            self.expand = Conv(self.inter_dim, int(1024 * multiplier), 3, 1)  # size is still
        elif level == 1:
            self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1)  # size still
            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)  # size is half
            self.expand = Conv(self.inter_dim, int(512 * multiplier), 3, 1)  # size is still
        elif level == 2:
            self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1)  # still
            self.compress_level_1 = Conv(int(512 * multiplier), self.inter_dim, 1, 1)  # still
            self.expand = Conv(self.inter_dim, int(256 * multiplier), 3, 1)  # still

        # when adding rfb, we use half number of channels to save memory # 这个地方以下的代码可以好好学一下，然后融会贯通
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(compress_c * 3, 3, 1, 1)  # channel is change, size is still
        self.vis = vis

    def forward(self, x):  # x为列表数据，[f1, f2, f3] channel:f1<f2<f3  size:f1>f2>f3
        """
        #
        256, 512, 1024
        from small -> large
        """
        x_level_0 = x[2]  # 最大特征层
        x_level_1 = x[1]  # 中间特征层
        x_level_2 = x[0]  # 最小特征层

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m=128, c_n=128, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)  # b,c_m,h,w
        B = self.convB(x)  # b,c_n,h,w
        V = self.convV(x)  # b,c_n,h,w
        tmpA = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1))
        attention_vectors = F.softmax(V.view(b, self.c_n, -1))
        # step 1: feature gating
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors)  # b,c_m,h*w
        tmpZ = tmpZ.view(b, self.c_m, h, w)  # b,c_m,h,w
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ
