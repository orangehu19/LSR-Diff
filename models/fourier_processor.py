import torch
import torch.nn as nn


class FourierHighFreqProcessor(nn.Module):
    def __init__(self, in_channels=6, out_channels=6, high_freq_ratio=0.25):
        """
        傅里叶变换高频信息处理器：提取高频特征并与空间域特征融合
        Args:
            in_channels: 输入通道数（条件图像+带噪图像=6）
            out_channels: 输出通道数（与输入一致，适配后续Mamba-UNet）
            high_freq_ratio: 高频区域占比（0.25=保留25%高频，可调）
        """
        super().__init__()
        self.high_freq_ratio = high_freq_ratio
        # 高频/空间域特征投影（确保维度对齐）
        self.high_freq_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.spatial_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 → [B, 6, H, W]
        Returns:
            融合高频的特征 → [B, 6, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 傅里叶变换（2D）+ 低频移至中心
        x_fft = torch.fft.fft2(x)
        x_fft_shift = torch.fft.fftshift(x_fft)
        
        # 2. 生成高频掩码（中心低频区域置0，保留边缘高频）
        radius = int(min(H, W) * (1 - self.high_freq_ratio) / 2)  # 低频区域半径
        center_h, center_w = H // 2, W // 2
        mask = torch.ones_like(x_fft_shift)
        mask[..., center_h-radius:center_h+radius, center_w-radius:center_w+radius] = 0
        
        # 3. 提取高频→逆傅里叶变换回空间域
        x_high_fft = x_fft_shift * mask
        x_high_shift = torch.fft.ifftshift(x_high_fft)
        x_high = torch.fft.ifft2(x_high_shift).real  # 取实部消除复数误差
        
        # 4. 高频与空间域特征融合（残差连接）
        x_high = self.high_freq_proj(x_high)
        x_spatial = self.spatial_proj(x)
        x_fused = x_spatial + x_high
        
        return x_fused