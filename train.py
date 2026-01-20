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

import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
from torch import nn
import copy


import warnings
# 强行过滤掉 torchvision 的过时警告，眼不见为净
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


# 尝试导入Tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# ==========================================================
# --- 借鉴参考代码思路：全局初始化 LPIPS，防止 NameError ---
# ==========================================================
try:
    from lpips import LPIPS

    # 设为全局变量，模块内所有函数均可访问
    lpips_fn = LPIPS(net='vgg').cuda()
except ImportError:
    lpips_fn = None


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             pre_densify_args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # --------------------------------- SaGPD 预密集化执行 ---------------------------------
    if pre_densify_args is not None and pre_densify_args['enabled']:
        try:
            print(f"[SaGPD] 正在执行预密集化校验...")
            gaussians.apply_SaGPD(
                scene=scene,
                pipe=pipe,
                background=background,
                knn_neighbors=pre_densify_args['knn_neighbors'],
                sparsity_threshold=pre_densify_args['sparsity_threshold'],
                opacity_scale=pre_densify_args['opacity_scale'],
                size_shrink=pre_densify_args['size_shrink'],
                perturb_strength=pre_densify_args['perturb_strength'],
                min_views=pre_densify_args['min_views'],
                depth_error=pre_densify_args['depth_error']
            )
        except Exception as e:
            print(f"SaGPD 运行出错: {e}")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="训练进度")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机抽取相机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_image = viewpoint_cam.original_image.cuda()

        # 渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, is_train=True, iteration=iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # 损失计算
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"损失": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 日志记录与评估
            training_report(dataset, tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[迭代 {}] 保存模型".format(iteration))
                scene.save(iteration)

            # 致密化
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, None)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步进
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[迭代 {}] 保存检查点".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("模型路径: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    return tb_writer


def training_report(args, tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 评估逻辑
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 仅评估测试集
        validation_configs = ({'name': '测试集', 'cameras': scene.getTestCameras()},)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg["render"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    # 计算 SSIM
                    ssim_test += ssim(image[None], gt_image[None]).item()
                    # 计算 LPIPS (使用全局变量 lpips_fn)
                    if lpips_fn is not None:
                        lpips_test += lpips_fn(image[None], gt_image[None]).item()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                # --- 借鉴点：直接在终端输出三个核心指标，简洁明了 ---
                print("\n[迭代 {}] {} 评估汇总: PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(
                    iteration, config['name'], psnr_test, ssim_test, lpips_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/ssim', ssim_test, iteration)
                    if lpips_fn:
                        tb_writer.add_scalar(config['name'] + '/lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="训练参数设置")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # --- SaGPD 命令行参数 ---
    parser.add_argument("--pre_densify", action="store_true")
    parser.add_argument("--pre_knn_neighbors", type=int, default=8)
    parser.add_argument("--pre_sparsity_threshold", type=float, default=0.7)
    parser.add_argument("--pre_opacity_scale", type=float, default=0.3)
    parser.add_argument("--pre_size_shrink", type=float, default=1.5)
    parser.add_argument("--pre_perturb_strength", type=float, default=0.1)
    parser.add_argument("--pre_min_consistency_views", type=int, default=2)
    parser.add_argument("--pre_depth_error_limit", type=float, default=0.01)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("开始优化: " + args.model_path)

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 封装 SaGPD 参数
    pre_densify_args = {
        'enabled': args.pre_densify,
        'knn_neighbors': args.pre_knn_neighbors,
        'sparsity_threshold': args.pre_sparsity_threshold,
        'opacity_scale': args.pre_opacity_scale,
        'size_shrink': args.pre_size_shrink,
        'perturb_strength': args.pre_perturb_strength,
        'min_views': args.pre_min_consistency_views,
        'depth_error': args.pre_depth_error_limit
    }

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, pre_densify_args)

    print("\n训练完成。")