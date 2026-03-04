import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2  
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
from utils.metrics import calculate_psnr, calculate_ssim  # 替换skimage，使用指定函数
from PIL import Image
import lpips
from pytorch_fid import fid_score
from torchvision import transforms


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Evaluate Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='LOLv2.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='/HUBU-AI093/hcc_24/model/DiffLL_OP/ckpt/260131_2batch_patch352_2down_fft_lolv2-real/model_val_best.pth.tar',
                        type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default='results/0131_lolv2_fft', type=str,
                        help="Location to save restored images")
    # parser.add_argument('--target_folder',
    #                     default='/HUBU-AI093/hcc_24/model/DiffLL_OP/data/Image_restoration/LL_dataset/LOLv1/val/high',
    #                     type=str,
    #                     help="Path to the folder containing target/reference images for PSNR/SSIM calculation")
    parser.add_argument('--target_folder',
                        default='/HUBU-AI093/hcc_24/model/DiffLL_OP/data/Image_restoration/LL_dataset/LOLv2-real/val/normal',
                        type=str,
                        help="Path to the folder containing target/reference images for PSNR/SSIM calculation")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def calculate_metrics(restored_folder, target_folder, device):
    """计算恢复图像与目标图像之间的PSNR、SSIM、LPIPS和FID（使用指定的calculate_psnr和calculate_ssim）"""
    if not os.path.exists(target_folder):
        print(f"错误: 目标图像文件夹 '{target_folder}' 不存在")
        return

    # 获取图像文件列表（仅保留图片格式）
    restored_files = sorted([f for f in os.listdir(restored_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    target_files = sorted([f for f in os.listdir(target_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(restored_files) == 0:
        print(f"错误: 在恢复图像文件夹 '{restored_folder}' 中未找到图像文件")
        return

    # 处理图像数量不匹配的情况
    if len(restored_files) != len(target_files):
        print(f"警告: 恢复图像数量 ({len(restored_files)}) 与目标图像数量 ({len(target_files)}) 不匹配")
        min_len = min(len(restored_files), len(target_files))
        restored_files = restored_files[:min_len]
        target_files = target_files[:min_len]
        print(f"将只计算前 {min_len} 对图像的指标")

    total_psnr = 0
    total_ssim = 0
    total_lpips = 0

    # 初始化LPIPS模型
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # LPIPS计算所需的图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("\n=== 图像质量评估结果 ===")
    for i, (restored_file, target_file) in enumerate(zip(restored_files, target_files)):
        # 读取图像（cv2读取，BGR格式）
        restored_path = os.path.join(restored_folder, restored_file)
        target_path = os.path.join(target_folder, target_file)
        res = cv2.imread(restored_path, cv2.IMREAD_COLOR)  # BGR格式
        gt = cv2.imread(target_path, cv2.IMREAD_COLOR)     # BGR格式

        # 检查图像是否读取成功
        if res is None or gt is None:
            print(f"警告: 无法读取图像 {restored_file} 或 {target_file}，跳过该对图像")
            continue

        # 确保图像尺寸一致（缩放目标图像到恢复图像尺寸）
        if res.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_CUBIC)

        # 计算PSNR和SSIM（使用指定函数，基于Y通道）
        cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
        cur_ssim = calculate_ssim(res, gt, test_y_channel=True)

        # 计算LPIPS（需转换为RGB格式）
        restored_rgb = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))  # BGR转RGB
        target_rgb = Image.fromarray(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))     # BGR转RGB

        # 预处理并计算LPIPS
        restored_tensor = transform(restored_rgb).unsqueeze(0).to(device)
        target_tensor = transform(target_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            img_lpips = loss_fn_vgg(restored_tensor, target_tensor).item()

        # 打印单张图像的指标
        print(f"图像 {i + 1}/{len(restored_files)}: {restored_file}")
        print(f"  PSNR: {cur_psnr:.4f} dB")
        print(f"  SSIM: {cur_ssim:.4f}")
        print(f"  LPIPS: {img_lpips:.4f}")

        # 累加指标
        total_psnr += cur_psnr
        total_ssim += cur_ssim
        total_lpips += img_lpips

    # 计算平均指标
    avg_psnr = total_psnr / len(restored_files) if restored_files else 0
    avg_ssim = total_ssim / len(restored_files) if restored_files else 0
    avg_lpips = total_lpips / len(restored_files) if restored_files else 0

    # 计算FID
    print("\n计算FID指标中...")
    fid_value = fid_score.calculate_fid_given_paths(
        [restored_folder, target_folder],
        batch_size=16,
        device=device,
        dims=2048
    )

    # 打印平均指标
    print("\n=== 平均指标 ===")
    print(f"平均 PSNR: {avg_psnr:.4f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"平均 LPIPS: {avg_lpips:.4f}")
    print(f"FID: {fid_value:.4f}")


def main():
    args, config = parse_args_and_config()

    # 设置运行设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # 数据加载
    print("=> using dataset '{}'".format(config.data.val_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders(parse_patches=True)

    # 创建模型
    print("=> creating denoising-diffusion model")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)

    # 加载模型权重
    checkpoint = torch.load(args.resume, map_location=device)
    model.diffusion.model.load_state_dict(checkpoint['state_dict'])

    # 执行恢复
    model.restore(val_loader)

    # 计算评估指标
    if args.target_folder:
        # calculate_metrics(os.path.join(args.image_folder, 'LSRW-Nikon'), args.target_folder, device)
        calculate_metrics(os.path.join(args.image_folder, 'LOLv2-real'), args.target_folder, device)


if __name__ == '__main__':
    main()