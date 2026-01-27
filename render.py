#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim

# 尝试导入 LPIPS
try:
    from lpipsPyTorch import lpips
except ImportError:
    try:
        from lpips import LPIPS

        # 兼容处理: 如果没有 lpipsPyTorch，使用 standard lpips 库封装一个函数
        _lpips_model = LPIPS(net='vgg').cuda()


        def lpips(img1, img2, net_type='vgg'):
            return _lpips_model(img1, img2)
    except ImportError:
        print("[Warning] LPIPS library not found. Metric will be 0.")


        def lpips(img1, img2, net_type='vgg'):
            return torch.tensor(0.0)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    PSNR = []
    SSIM = []
    LPIPS_val = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        PSNR.append(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)))
        SSIM.append(ssim(rendering.unsqueeze(0), gt.unsqueeze(0)))
        LPIPS_val.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type='vgg'))

    psnr_mean = torch.tensor(PSNR).mean().item()
    ssim_mean = torch.tensor(SSIM).mean().item()
    lpips_mean = torch.tensor(LPIPS_val).mean().item()

    print(f'\n[Evaluation] {name} Set Results:')
    print(f'PSNR : {psnr_mean:>12.7f}')
    print(f'SSIM : {ssim_mean:>12.7f}')
    print(f'LPIPS : {lpips_mean:>12.7f}')

    # ==========================================================
    # [核心修改] 继承 train.py 写入的点数信息
    # ==========================================================
    metrics_file = os.path.join(model_path, 'metrics_{}.txt'.format(iteration))

    # 暂存旧数据
    existing_lines = []
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            # 只保留包含 Points 的行
            existing_lines = [line for line in f.readlines() if "Points" in line]

    # 重新写入（覆盖模式，但把旧数据写回去）
    with open(metrics_file, 'w') as f:
        # 1. 写入新的渲染指标
        f.write(f'PSNR : {psnr_mean:>12.7f}\n')
        f.write(f'SSIM : {ssim_mean:>12.7f}\n')
        f.write(f'LPIPS : {lpips_mean:>12.7f}\n')

        # 2. 写回点数信息 (如果存在)
        for line in existing_lines:
            f.write(line)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)