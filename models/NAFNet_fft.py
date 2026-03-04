import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class OurTokenMixer_For_Local(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(OurTokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim
        self.CDilated = nn.Sequential(nn.BatchNorm2d(self.dim_sp),
                                     nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, groups=self.dim_sp)
                                        )

    def forward(self, x):
        x = self.CDilated(x)  # (batch, c*2, h, w/2+1)

        return x


class OurTokenMixer_For_Global(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(OurTokenMixer_For_Global, self).__init__()
        self.dim = dim
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1)
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.FFC(x)
        x = self.conv_fina(x)

        return x


class OurMixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_local=OurTokenMixer_For_Local,
            token_mixer_for_global=OurTokenMixer_For_Global
    ):
        super(OurMixer, self).__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim=self.dim)
        self.mixer_global = token_mixer_for_global(dim=self.dim)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU()
        )


    def forward(self, x):
        x = self.conv_init(x)
        x = self.mixer_global(x)
        x = self.gelu(x)
        x = self.ca_conv(x)

        return x


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, padding=0, groups=self.groups, bias=True),
            nn.GELU(),
        )

    def forward(self, x):
        batch, c, h, w = x.size()
        # 1. 进行 FFT 并分离实部和虚部
        ffted = torch.fft.rfft2(x, norm='ortho')  # 输出形状: [B, C, H, W//2 + 1]（复数张量）
        real = ffted.real  # 实部: [B, C, H, W//2 + 1]
        imag = ffted.imag  # 虚部: [B, C, H, W//2 + 1]

        # 2. 拼接实部和虚部，调整维度为 [B, C*2, H, W//2 + 1]
        # 关键修正：先在通道维度拼接，再确保连续
        fft_cat = torch.cat([real, imag], dim=1).contiguous()  # 拼接后立即确保连续

        # 3. 卷积处理
        fft_out = self.conv(fft_cat)  # 输出形状: [B, out_channels*2, H, W//2 + 1]

        # 4. 拆分实部和虚部，并调整维度为复数张量要求的形状
        # 拆分后形状: [B, out_channels, H, W//2 + 1] 各两个
        real_out, imag_out = torch.chunk(fft_out, 2, dim=1)
        # 合并为 [B, out_channels, H, W//2 + 1, 2]，并确保最后一维步长为 1
        fft_combined = torch.stack([real_out, imag_out], dim=-1).contiguous()  # 关键：stack 后 contiguous

        # 5. 转换为复数张量并执行逆 FFT
        fft_complex = torch.view_as_complex(fft_combined)  # 此时最后一维步长为 1
        output = torch.fft.irfft2(fft_complex, s=(h, w), norm='ortho')  # 逆变换回空域

        return output

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.dim = c
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True))
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.mixer = OurMixer(dim=self.dim)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        
        
        # 新增：整合PW-FNet的FFT模块到FFN中
        self.fourier_unit = FourierUnit(c, c)

    def forward(self, inp):
        # x = inp
        # x = self.norm1(x)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.sg(x)
        # x = x * self.sca(x)
        # x = self.conv3(x)
        # x = self.dropout1(x)
        # y = inp + x * self.beta

        x = inp
        x = self.norm1(x)
        x = self.mixer(x)
        y = x * self.beta + inp

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        # 插入FFT频域处理
        x_fft = self.fourier_unit(x)
        x = x + x_fft  # 残差连接融合频域特征
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma
        

class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_size = train_size
        self.fast_imp = fast_imp

if __name__ == '__main__':
    img_channel = 3
    width = 32
    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    tensor_randn = torch.randn(16, 3, 256, 256)
    out = net(tensor_randn)
    print(out.shape)