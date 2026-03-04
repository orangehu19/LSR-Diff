import time

import torch
import numpy as np
import utils
import os
import torch.nn.functional as F


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        os.makedirs(image_folder, exist_ok=True)  # 创建保存目录

        inference_times = []  # 存储每张图片的推理时间

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                # 记录开始时间（包括数据准备）
                start_time = time.time()

                # x_cond = x[:, :3, :, :].to(self.diffusion.device)
                # b, c, h, w = x_cond.shape
                # img_h_32 = int(32 * np.ceil(h / 32.0))
                # img_w_32 = int(32 * np.ceil(w / 32.0))
                # x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                # print(x_cond.shape)
                b, c, h, w = x_cond.shape
                img_h_32 = int(64 * np.ceil(h / 64.0))
                img_w_32 = int(64 * np.ceil(w / 64.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                # x_cond = F.pad(x_cond, (0, 40, 0, -16), 'reflect')
                # x_cond = F.pad(x_cond, (0, 0, 0, -16), 'reflect')   #建筑集使用
                # print(x_cond.shape)
                # x_output = self.diffusive_restoration(x_cond)


                # 记录模型推理开始时间
                model_start_time = time.time()

                x_output = self.diffusive_restoration(x_cond)
                # x_output, x_LL = self.diffusive_restoration(x_cond)

                # 记录模型推理结束时间
                model_end_time = time.time()

                # 检查x_output是否为元组
                if isinstance(x_output, tuple):
                    # 假设第一个元素是输出图像
                    x_output = x_output[0]

                x_output = x_output[:, :, :h, :w]
                # utils.logging.save_image(x_LL, os.path.join(image_folder, f"{y[0]}_LL.png"))
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))
                

                # 计算单张图片推理时间（仅模型推理部分）
                inference_time = model_end_time - model_start_time
                inference_times.append(inference_time)


                # 打印进度和时间信息
                print(f"处理图片 {y[0]} | 推理时间: {inference_time:.4f}s")

        # 计算统计信息
        if inference_times:
            avg_time = sum(inference_times) / len(inference_times)
            total_time = sum(inference_times)
            print(f"\n=== 推理时间统计 ===")
            print(f"总处理图片数: {len(inference_times)}")
            print(f"平均推理时间: {avg_time:.4f}s/张")
            print(f"总推理时间: {total_time:.4f}s")
            print(f"吞吐量: {len(inference_times) / total_time:.2f}张/秒")

        return inference_times  # 返回推理时间列表


    def diffusive_restoration(self, x_cond):
        x_output = self.diffusion.model(x_cond)
        return x_output["pred_x"]
        # return x_output["pred_x"], x_output["pred_LL"]

