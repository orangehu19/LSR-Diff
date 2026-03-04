import torch
import torch.nn as nn
import pywt
import torchvision.transforms as transforms
def Normalize(x):
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(inp, oup, kernel_size=1),
        #     nn.GroupNorm(8, oup),
        #     nn.GELU()
        # )
        # self.attention = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(oup, oup // 16, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(oup // 16, oup, kernel_size=1),
        #     nn.Sigmoid()
        # )

        mip = max(18, inp // groups)
        self.conv0=nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # print(x.shape)
        identity = self.conv0(x)
        # identity = x
        # print(identity.shape)
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        # y = identity * x_w * x_h
        y = identity * x_w+ identity * x_h

        return y

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

#近似系数的融合小波，实验结果表明实际有效，但有网格问题
def dwt_init_new(x):
    db4 = pywt.wavedec2(x.cpu().detach().numpy(), 'db1', level=1)[0]
    db4=torch.Tensor(db4).to('cuda:0')
    haar = pywt.wavedec2(x.cpu().detach().numpy(), 'haar', level=1)[0]
    haar = torch.Tensor(haar).to('cuda:0')
    bior = pywt.wavedec2(x.cpu().detach().numpy(), 'bior1.1', level=1)[0]
    bior = torch.Tensor(bior).to('cuda:0')
    sym = pywt.wavedec2(x.cpu().detach().numpy(), 'sym2', level=1)[0]
    sym = torch.Tensor(sym).to('cuda:0')
    sym = sym[:, :, 0:sym.shape[2] - 1, 0:sym.shape[3] - 1]
    return torch.cat((db4, haar, bior, sym), 0)


def dwt_init_merge(x):
    # 近似系数
    db4 = pywt.wavedec2(x.cpu().detach().numpy(), 'db1', level=1)[0]
    db4 = torch.Tensor(db4).to('cuda:0')
    haar = pywt.wavedec2(x.cpu().detach().numpy(), 'haar', level=1)[0]
    bior = pywt.wavedec2(x.cpu().detach().numpy(), 'bior1.1', level=1)[0]
    sym = pywt.wavedec2(x.cpu().detach().numpy(), 'sym2', level=1)[0]
    sym = sym[:, :, 0:bior.shape[2], 0:bior.shape[3]]
    #细节系数
    db4_1 = pywt.wavedec2(x.cpu().detach().numpy(), 'db1', level=1)[1][0]
    db4_1 = torch.Tensor(db4_1).to('cuda:0')
    haar_1 = pywt.wavedec2(x.cpu().detach().numpy(), 'haar', level=1)[1][0]
    haar = torch.Tensor(haar).to('cuda:0')
    haar_1 = torch.Tensor(haar_1).to('cuda:0')
    bior_1 = pywt.wavedec2(x.cpu().detach().numpy(), 'bior1.1', level=1)[1][0]
    bior = torch.Tensor(bior).to('cuda:0')
    bior_1 = torch.Tensor(bior_1).to('cuda:0')
    # print(bior_1.shape)
    sym_1 = pywt.wavedec2(x.cpu().detach().numpy(), 'sym2', level=1)[1][0]
    sym = torch.Tensor(sym).to('cuda:0')
    # print(sym.shape)
    sym_1 = torch.Tensor(sym_1).to('cuda:0')
    sym_1 = sym_1[:, :, 0:bior.shape[2], 0:bior.shape[3]]
    # print(sym_1.shape)
    merge1=torch.cat((db4, haar, bior, sym), 0)
    # merge1=torch.Tensor(merge1).to('cuda:0')
    merge2=torch.cat((db4_1, haar_1, bior_1, sym_1), 0)
    # merge2=torch.Tensor(merge2).to('cuda:0')

    merge=torch.cat((merge1,merge2),1)
    conv1 = torch.nn.Conv2d(6,
                           12,
                           kernel_size=1,
                           stride=1,
                           padding=0).cuda()
    conv2 = torch.nn.Conv2d(12,
                           3,
                           kernel_size=1,
                           stride=1,
                           padding=0).cuda()
    merge=conv1(merge)
    merge=conv2(merge)
    # print('merge1',merge1.shape)
    # print('merge2',merge2.shape)
    return merge

# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    # print('haar',x.shape,h.shape)

    return h

import torch.nn as nn

class MultiBranchDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiBranchDownsample, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        fused = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        return self.fusion(fused)

# 创建多分支下采样模块
downsample_module = MultiBranchDownsample(in_channels=3, out_channels=3)

class Bicubic_plus_plus(nn.Module):
    def __init__(self, sr_rate=2):
        super(Bicubic_plus_plus, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv_out = nn.Conv2d(32, (2*sr_rate)**2 * 3, kernel_size=3, padding=1, bias=False)
        self.Depth2Space = nn.PixelShuffle(2*sr_rate)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.1)


    def forward(self, x):
        # xx=self.transform(x)
        x0 = self.conv0(x)
        x0 = self.act(x0)
        x1 = self.conv1(x0)
        x1 = self.act(x1)
        x2 = self.conv2(x1)
        x2 = self.act(x2) + x0
        y = self.conv_out(x2)
        y = self.Depth2Space(y)
        return y

# 使用哈尔 haar 小波变换来实现二维离散小波
def db_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=1 / 2):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel number of a convolution branch

        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)

        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)

        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.split_indexes = (gc, gc, gc, in_channels - 3 * gc)
        self.down_conv=nn.Conv2d(3,3, kernel_size=3, stride=2,padding=1)

    def forward(self, x):
        # B, C, H, W = x.shape
        x_hw, x_w, x_h, x_id = torch.split(x, self.split_indexes, dim=1)
        x=torch.cat(
            (self.dwconv_hw(x_hw),
             self.dwconv_w(x_w),
             self.dwconv_h(x_h),
             x_id),
            dim=1)
        # x_hw, x_w, x_h, x_id = torch.split(x, self.split_indexes, dim=1)
        # x = torch.cat(
        #     (self.dwconv_hw(x_hw),
        #      self.dwconv_w(x_w),
        #      self.dwconv_h(x_h),
        #      x_id),
        #     dim=1)
        x=self.down_conv(x)

        return x

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class WT_s(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)