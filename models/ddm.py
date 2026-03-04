# Unet+IDC+SR+FFT

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# from pywt import dwt
import tqdm
# from modelscope.models.cv.cartoon.loss import content_loss

import utils
from models.BSRN_arch import BSRN
# from models.esc import ESC
from models.unet import DiffusionUNet
from models.wavelet import DWT, IWT, Bicubic_plus_plus, CoordAtt, InceptionDWConv2d
from pytorch_msssim import ssim
# from models.mods import HFRM
# import pywt
import torchvision.transforms as transforms
from models.NAFNet_fft import NAFBlock

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas
#
# class DilatedConvModel(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation):
#         super(DilatedConvModel, self).__init__()
#         # 定义空洞卷积层
#         self.dilated_conv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=1,  # 保持空间维度不变
#             padding=dilation,  # 确保输出大小与输入相同
#             dilation=dilation  # 设置扩张率
#         )
#
#     def forward(self, x):
#         return self.dilated_conv(x)
#
#
# class ProgressiveDownsample(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, blur_kernel_size=5,
#                  blur_sigma=1.0):
#         super(ProgressiveDownsample, self).__init__()
#
#         # 定义卷积层
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
#
#         # 定义抗锯齿滤波器
#         self.blur = transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
#
#     def forward(self, x):
#         # 先应用抗锯齿滤波器
#         x_blurred = self.blur(x)
#
#         # 再应用卷积层进行下采样
#         x_downsampled = self.conv(x_blurred)
#
#         return x_downsampled
class Edge(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel kernel (x and y). 会自动扩展到多个通道
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sx', sobel_x)
        self.register_buffer('sy', sobel_y)

    def forward(self, pred):
        # pred, gt: [B, C, H, W], 假设 C=3
        # 按通道做 conv 后求和
        b, c, h, w = pred.shape
        sx = self.sx.repeat(c, 1, 1, 1)
        sy = self.sy.repeat(c, 1, 1, 1)
        # group conv 实现按通道卷积
        pred_gx = F.conv2d(pred, sx, padding=1, groups=c)
        pred_gy = F.conv2d(pred, sy, padding=1, groups=c)
        pred_edge = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)

        return pred_edge

class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device
        self.Unet = DiffusionUNet(config)
        self.down_conv = InceptionDWConv2d(3)
        self.bicu=BSRN()
        self.deblur=NAFBlock(3)
        self.edge1=Edge()
        # self.conv1=torch.nn.Conv2d(6,3,kernel_size=3,padding=1)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            # et = self.Unet((x_cond+xt), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, x):
        data_dict = {}
        dwt, idwt = DWT(), IWT()
        # bicu= Bicubic_plus_plus().cuda()
        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)
        input_high0 = input_img_norm
        input_LL_LL = self.down_conv(input_high0)
        # input_LL_LL = self.down_conv(input_LL_LL1)
        b = self.betas.to(input_img.device)
        t = torch.randint(low=0, high=self.num_timesteps, size=(input_LL_LL.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_LL_LL.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        e = torch.randn_like(input_LL_LL)
        if self.training:
            # print(1111)
            gt_img=data_transform(x[:, 3:, :, :])
            gt_LL_LL = F.interpolate(gt_img, scale_factor=0.5, mode='bicubic', align_corners=False)
            # gt_LL_LL = data_transform(gt_LL_LL)
            gt_SR = self.bicu(gt_LL_LL)
            x = gt_LL_LL * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([input_LL_LL, x], dim=1), t.float())
            denoise_LL_LL = self.sample_training(input_LL_LL, b)
            pred_LL = inverse_data_transform(denoise_LL_LL)
            # print('1',pred_LL.shape)

            pred_SR=self.bicu(input_LL_LL)
            pred_x=self.bicu(pred_LL)
            # print('2', pred_x.shape)
            pred_x = self.deblur(pred_x)
            # pred_x = inverse_data_transform(pred_x)
            data_dict["input_high0"] = input_high0
            data_dict["gt_SR"] = gt_SR
            data_dict["pred_LL"] = pred_LL
            data_dict["pred_SR"] = pred_SR
            data_dict["gt_LL"] = gt_LL_LL
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e
            # data_dict["SR"]=bicu(input_LL_LL)

        else:
            denoise_LL_LL = self.sample_training(input_LL_LL, b)
            # pred_LL = denoise_LL_LL
            pred_LL = inverse_data_transform(denoise_LL_LL)
            pred_x=  self.bicu(pred_LL)
            pred_x = self.deblur(pred_x)
            # pred_x = inverse_data_transform(pred_x)
            data_dict["pred_x"] = pred_x
            data_dict["pred_LL"] = pred_LL

        return data_dict

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel kernel (x and y). 会自动扩展到多个通道
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sx', sobel_x)
        self.register_buffer('sy', sobel_y)

    def forward(self, pred, gt):
        # pred, gt: [B, C, H, W], 假设 C=3
        # 按通道做 conv 后求和
        b, c, h, w = pred.shape
        sx = self.sx.repeat(c, 1, 1, 1)
        sy = self.sy.repeat(c, 1, 1, 1)
        # group conv 实现按通道卷积
        pred_gx = F.conv2d(pred, sx, padding=1, groups=c)
        pred_gy = F.conv2d(pred, sy, padding=1, groups=c)
        gt_gx = F.conv2d(gt, sx, padding=1, groups=c)
        gt_gy = F.conv2d(gt, sy, padding=1, groups=c)
        pred_edge = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)
        gt_edge = torch.sqrt(gt_gx ** 2 + gt_gy ** 2 + 1e-6)
        return F.l1_loss(pred_edge, gt_edge)

# 在 __init__ 中添加
from torchvision.models import vgg16
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16].eval().cuda()  # 第3池化层前
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        x_feat = self.vgg(x)
        y_feat = self.vgg(y)
        return self.loss(x_feat, y_feat)

# 初始化

class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        # self.bicu = Bicubic_plus_plus()
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.edge_loss=EdgeLoss().cuda()
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()
        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        self.datetime = "260203_2batch_patch256_2down_fft_lsrw"

        # ========== 新增：初始化验证相关变量 ==========
        self.val_image_folder = os.path.join(args.image_folder, "validation_results")
        self.metrics_file = os.path.join(config.data.ckpt_dir, self.datetime, "training_metrics.txt")
        self.best_val_psnr = 0.0
        self.best_val_ssim = 0.0
        
        os.makedirs(self.val_image_folder, exist_ok=True)
        os.makedirs(os.path.join(config.data.ckpt_dir, self.datetime), exist_ok=True)
        # 初始化日志文件（写入表头）
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                f.write("Epoch,Noise_Loss,Photo_Loss,Frequency_Loss,PSNR,SSIM,Learning_Rate\n")

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def write_metrics(self, epoch, noise_loss, photo_loss, frequency_loss, psnr=0.0, ssim=0.0):
        """
        写入训练/验证指标到日志文件，确保实时保存
        Args:
            epoch: 当前epoch
            noise_loss/photo_loss/frequency_loss: 训练损失
            psnr/ssim: 验证指标（默认0.0，训练时可不传）
        """
        # 获取当前学习率
        lr = self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
        # 拼接日志行（保留4位小数）
        log_line = f"{epoch},{noise_loss:.4f},{photo_loss:.4f},{frequency_loss:.4f},{psnr:.4f},{ssim:.4f},{lr:.6f}\n"
        
        # 关键：以追加模式+行缓冲打开文件，写入后自动刷新
        with open(self.metrics_file, 'a', encoding='utf-8', buffering=1) as f:
            f.write(log_line)

    def calculate_val_metrics(self, pred, gt):
        """计算验证集PSNR和SSIM"""
        pred = torch.clamp(pred, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)
        mse = F.mse_loss(pred, gt, reduction='mean')
        psnr = 10 * torch.log10(1.0 / mse)
        ssim_val = ssim(pred, gt, data_range=1.0, size_average=True)
        return psnr.item(), ssim_val.item()

    # def train(self, DATASET):
    #     psnr_line = 1
    #     cudnn.benchmark = True
    #     train_loader, val_loader = DATASET.get_loaders()

    #     if os.path.isfile(self.args.resume):
    #         self.load_ddm_ckpt(self.args.resume)

    #     for epoch in range(self.start_epoch, self.config.training.n_epochs):
    #         print('\nepoch: ', epoch+1)
    #         # print('loss', max)
    #         data_start = time.time()
    #         data_time = 0
    #         for i, (x, y) in enumerate(train_loader):
    #             x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
    #             data_time += time.time() - data_start
    #             self.model.train()
    #             self.step += 1
    #             x = x.to(self.device)
    #             output = self.model(x)
    #             noise_loss, photo_loss, frequency_loss = self.estimation_loss(x, output)
    #             loss = noise_loss + photo_loss + frequency_loss
    #             # if self.step % 10 == 0:
    #             #     print("step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, "
    #             #           "frequency_loss:{:.4f}".format(self.step, self.scheduler.get_last_lr()[0],
    #             #                                          noise_loss.item(), photo_loss.item(),
    #             #                                          frequency_loss.item()))
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #             self.ema_helper.update(self.model)
    #             # self.ema_helper.update(self.bicu)
    #             # data_start = time.time()
    #         psnr1 = self.sample_validation_patches(val_loader, epoch, self.step)
    #         if psnr1 > psnr_line:
    #             # max = loss
    #             psnr_line = psnr1

    #             self.model.eval()
    #             print('save_best_epoch_weights',epoch,psnr1)
    #             # self.sample_validation_patches(val_loader, self.step)
    #             utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
    #                                            'state_dict': self.model.state_dict(),
    #                                            'optimizer': self.optimizer.state_dict(),
    #                                            'scheduler': self.scheduler.state_dict(),
    #                                            'ema_helper': self.ema_helper.state_dict(),
    #                                            'params': self.args,
    #                                            'config': self.config},
    #                                           filename=os.path.join(self.config.data.ckpt_dir,
    #                                                                 'model_best_train'))
    #         self.scheduler.step()
    def train(self, DATASET):
        min_train_loss = 999
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('\nepoch: ', epoch+1)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                output = self.model(x)

                noise_loss, photo_loss, frequency_loss = self.estimation_loss(x, output)

                loss = abs(noise_loss + photo_loss + frequency_loss)
                # loss = noise_loss + photo_loss + frequency_loss
                if self.step % 1 == 0:
                    print("step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, "
                          "frequency_loss:{:.4f}".format(self.step, self.scheduler.get_last_lr()[0],
                                                         noise_loss.item(), photo_loss.item(),
                                                         frequency_loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                # 添加全局梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, epoch, self.step)

                    utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.config.data.ckpt_dir,self.datetime, 'model_latest'))
                    
                    self.write_metrics(
                        epoch=epoch+1,  # 当前epoch（+1是因为epoch从0开始）
                        noise_loss=noise_loss.item(),
                        photo_loss=photo_loss.item(),
                        frequency_loss=frequency_loss.item()
                    )

                # 保存训练损失最优权重
                if loss < min_train_loss:
                    min_train_loss = loss.item()
                    self.model.eval()
                    # print(f"New Best Train Loss: {min_train_loss:.6f}, Saving weights...")
                    utils.logging.save_checkpoint(
                        {
                            'step': self.step,
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                            'ema_helper': self.ema_helper.state_dict(),
                            'params': self.args,
                            'config': self.config
                        },
                        filename=os.path.join(self.config.data.ckpt_dir, self.datetime, 'model_train_best')
                    )
                        
            self.scheduler.step()

    def estimation_loss(self, x, output):

        # input_high0, input_high1, gt_high0, gt_high1 = output["input_high0"], output["input_high1"],\
        #                                                output["gt_high0"], output["gt_high1"]
        input_high0 = output["input_high0"]
        gt_SR=output["gt_SR"]
        pred_SR=output["pred_SR"]
        gt_LL = output['gt_LL']
        pred_LL = output["pred_LL"]

        pred_x, noise_output, e = output["pred_x"], output["noise_output"], output["e"]
        gt_img = x[:, 3:, :, :].to(self.device)
        # print(gt_img)
        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)
        # print(pred_x.shape)
        # print(gt_img.shape)
        frequency_loss = self.l1_loss(pred_x, gt_img)+self.TV_loss(pred_x)+0.5*self.l1_loss(pred_LL, gt_LL)
        mse = F.mse_loss(pred_x, gt_img, reduction='mean')
        # 计算 PSNR
        psnr_loss =1 /(10 * torch.log10(1.0 / mse))
        content_loss=self.l2_loss(pred_x, gt_img)
        # print(10 * torch.log10((1 ** 2) / mse))
        # l1_loss = self.l1_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0)
        perceptual_loss = self.perceptual_loss(pred_x, gt_img)
        SR_loss = 0.5 * self.l1_loss(gt_SR, gt_img)
        edge_loss = self.edge_loss(pred_x, gt_img)
        # 3. 组合 Photo Loss
        photo_loss = (
                1.0 * content_loss +
                1.0 * ssim_loss +
                0.5 * perceptual_loss +
                0.5 * edge_loss+
                1.0*SR_loss+
                1.0*psnr_loss
        )
        
        return noise_loss, photo_loss, frequency_loss

    def calculate_val_metrics(self, pred, gt):
        """计算验证集PSNR和SSIM"""
        pred = torch.clamp(pred, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)
        mse = F.mse_loss(pred, gt, reduction='mean')
        psnr = 10 * torch.log10(1.0 / mse)
        ssim_val = ssim(pred, gt, data_range=1.0, size_average=True)
        return psnr.item(), ssim_val.item()

    # def sample_validation_patches(self, val_loader,epoch,step):
    #     print('start_val:')
    #     image_folder = os.path.join(self.args.image_folder, self.config.data.type + str(self.config.data.patch_size))
    #     self.model.eval()
    #     total_psnr = 0.0
    #     total_ssim = 0.0
    #     num_samples = 0
    #     epoch = epoch

    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(val_loader):
    #             # x1=random.randint(0,248)
    #             # y1=random.randint(0,48)
    #             # x_input=x[:, :3, x1:x1+352, y1:y1+352]
    #             b, _, img_h, img_w = x.shape
    #             num_samples += b

    #             # 尺寸补齐到32的倍数
    #             img_h_32 = int(32 * np.ceil(img_h / 32.0))
    #             img_w_32 = int(32 * np.ceil(img_w / 32.0))

    #             x_padded = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')
    #             # print(x_padded.shape)

    #             # 模型推理
    #             out = self.model(x_padded.to(self.device))
    #             pred_x = out["pred_x"]
    #             pred_x = pred_x[:, :3, :img_h, :img_w]  # 裁剪回原始尺寸

    #             # 提取GT图像
    #             gt_img = x[:, 3:,:,:].to(self.device)

    #             # 计算单批指标
    #             batch_psnr, batch_ssim = self.calculate_val_metrics(pred_x, gt_img)
    #             total_psnr += batch_psnr * b
    #             total_ssim += batch_ssim * b

    #             # 保存验证图像
    #             # step_folder = os.path.join(self.val_image_folder, f"step_{step}")
    #             # os.makedirs(step_folder, exist_ok=True)
    #             # for idx in range(b):
    #             #     img_name = f"{y[idx]}.png"
    #             #     utils.logging.save_image(pred_x[idx].cpu(), os.path.join(step_folder, img_name))

    #     # 计算平均指标
    #     avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0
    #     avg_ssim = total_ssim / num_samples if num_samples > 0 else 0.0
    #     print(f" Avg PSNR: {avg_psnr:.4f} dB | Avg SSIM: {avg_ssim:.4f}")
    #     return  avg_psnr

    def sample_validation_patches(self, val_loader, epoch, step):
        """执行验证集评估，返回平均PSNR和SSIM"""
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        num_samples = 0
        epoch = epoch

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x=x[:, :, :, :]
                b, _, img_h, img_w = x.shape
                num_samples += b
                # print('input:',x.shape)

                # 尺寸补齐到32的倍数
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x_padded = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')
                # x_input = F.pad(x, (0, 40, 0, 112), 'reflect')

                # 模型推理
                out = self.model(x_padded.to(self.device))
                pred_x = out["pred_x"]
                # print('out',pred_x.shape)
                # print('out:',pred_x.shape)
                pred_x = pred_x[:, :, :img_h, :img_w]  # 裁剪回原始尺寸
                # print('pred_X',pred_x.shape)
                # pred_x = pred_x[:, :, :384, :384]
                # pred_x = pred_x[:, :, :256, :256]  # 裁剪回原始尺寸

                # 提取GT图像
                gt_img = x[:, 3:, :, :].to(self.device)
                # print('gt',gt_img.shape)
                # gt_img = x[:, 3:, :256, :256].to(self.device)
                # gt_img = x[:, 3:, :384, :384].to(self.device)
                # print('gt:',gt_img.shape)

                # 计算单批指标
                batch_psnr, batch_ssim = self.calculate_val_metrics(pred_x, gt_img)
                total_psnr += batch_psnr * b
                total_ssim += batch_ssim * b

                # # 保存验证图像
                # step_folder = os.path.join(self.val_image_folder, f"step_{step}")
                # os.makedirs(step_folder, exist_ok=True)
                # for idx in range(b):
                #     img_name = f"{y[idx]}.png"
                #     utils.logging.save_image(pred_x[idx].cpu(), os.path.join(step_folder, img_name))

        # 计算平均指标
        avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0
        avg_ssim = total_ssim / num_samples if num_samples > 0 else 0.0
        print(f"Validation Epoch:{epoch+1} | Avg PSNR: {avg_psnr:.4f} dB | Avg SSIM: {avg_ssim:.4f}")

        # 保存最优验证权重
        if avg_psnr > self.best_val_psnr or (avg_psnr == self.best_val_psnr and avg_ssim > self.best_val_ssim):
            self.best_val_psnr = avg_psnr
            self.best_val_ssim = avg_ssim
            print(f"New Best Validation! Saving weights...")
            utils.logging.save_checkpoint(
                {
                    # 'step': step,
                    # 'epoch': self.start_epoch + (step // len(val_loader)),
                    # 'state_dict': self.model.state_dict(),
                    # 'ema_helper': self.ema_helper.state_dict(),
                    # 'val_psnr': avg_psnr,
                    # 'val_ssim': avg_ssim
                    'step': self.step, 'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'ema_helper': self.ema_helper.state_dict(),
                    'params': self.args,
                    'config': self.config
                },
                filename=os.path.join(self.config.data.ckpt_dir, self.datetime, 'model_val_best')
            )

        return avg_psnr, avg_ssim