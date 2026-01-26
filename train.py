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
from torchvision.utils import save_image
from torch import nn
import copy
import warnings

# 过滤 torchvision 的过时警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# 尝试导入 Tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 全局初始化 LPIPS 评估器
try:
    from lpips import LPIPS

    lpips_fn = LPIPS(net='vgg').cuda()
except ImportError:
    lpips_fn = None

from arguments import ModelParams, PipelineParams, OptimizationParams


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             pre_densify_args):
    # 用于记录最后一次评估的指标
    final_metrics = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}

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

    # --- SaGPD / aGPD-Lite++ 预密集化执行 ---
    # [同步修改] 使用字典解包传递所有参数 (包括新增的高级参数)
    if pre_densify_args is not None and pre_densify_args['enabled']:
        try:
            print(f"[SaGPD] 正在执行预密集化校验...")
            gaussians.apply_SaGPD(
                scene=scene,
                pipe=pipe,
                background=background,
                # 使用字典推导式过滤掉 'enabled' 键，其余全部传给函数
                **{k: v for k, v in pre_densify_args.items() if k != 'enabled'}
            )
        except Exception as e:
            # 捕获异常防止中断训练，但打印完整错误栈以便调试
            import traceback
            traceback.print_exc()
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

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_image = viewpoint_cam.original_image.cuda()

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, is_train=True, iteration=iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

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

            # 日志记录与评估：获取评估指标
            if iteration in testing_iterations:
                res = training_report(dataset, tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                      testing_iterations, scene, render, (pipe, background))
                if res is not None:
                    final_metrics = res

            if (iteration in saving_iterations):
                print("\n[迭代 {}] 保存模型".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, None)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[迭代 {}] 保存检查点".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # 返回最终捕获的评估指标
    return final_metrics


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
    metrics_to_return = None

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
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
                    ssim_test += ssim(image[None], gt_image[None]).item()
                    if lpips_fn is not None:
                        lpips_test += lpips_fn(image[None], gt_image[None]).item()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])

                print("\n[迭代 {}] {} 评估汇总: PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(
                    iteration, config['name'], psnr_test, ssim_test, lpips_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/ssim', ssim_test, iteration)
                    if lpips_fn:
                        tb_writer.add_scalar(config['name'] + '/lpips', lpips_test, iteration)

                # 记录指标用于返回
                metrics_to_return = {"psnr": psnr_test, "ssim": ssim_test, "lpips": lpips_test}

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return metrics_to_return


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

    # --- SaGPD / aGPD-Lite++ 参数设置 (Full Control) ---
    parser.add_argument("--pre_densify", action="store_true", help="启用 SaGPD 预密集化")
    parser.add_argument("--pre_knn_neighbors", type=int, default=16, help="KNN邻居数 K (默认: 16)")
    parser.add_argument("--pre_sparsity_threshold", type=float, default=0.7, help="稀疏度分位数阈值 tau_s (默认: 0.7)")
    parser.add_argument("--pre_opacity_scale", type=float, default=0.3, help="不透明度系数 gamma_o (默认: 0.3)")
    parser.add_argument("--pre_size_shrink", type=float, default=1.5, help="尺寸收缩系数 delta (默认: 1.5)")
    parser.add_argument("--pre_min_consistency_views", type=int, default=2, help="最小一致性视角 M (默认: 2)")

    # [新增] 高级调参参数 (Advanced Tuning Args) - 同步自 apply_SaGPD
    parser.add_argument("--pre_dt_quantile", type=float, default=0.6, help="长边判断分位数 dt (默认: 0.6)")
    parser.add_argument("--pre_len_threshold_mult", type=float, default=2.0, help="边长阈值倍数 (默认: 2.0)")
    parser.add_argument("--pre_align_ql", type=float, default=0.1, help="DPT对齐下分位 (默认: 0.1)")
    parser.add_argument("--pre_align_qh", type=float, default=0.9, help="DPT对齐上分位 (默认: 0.9)")
    parser.add_argument("--pre_eta_o_quantile", type=float, default=0.9, help="动态误差阈值分位数 eta_o (默认: 0.9)")
    parser.add_argument("--pre_ratio_clamp_min", type=float, default=0.5, help="几何纠偏最小比例 (默认: 0.5)")
    parser.add_argument("--pre_ratio_clamp_max", type=float, default=2.0, help="几何纠偏最大比例 (默认: 2.0)")
    parser.add_argument("--pre_visible_count_threshold", type=int, default=50, help="对齐最少可见点数 (默认: 50)")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("开始优化: " + args.model_path)
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    pre_densify_args = {
        'enabled': args.pre_densify,
        'knn_neighbors': args.pre_knn_neighbors,
        'sparsity_threshold': args.pre_sparsity_threshold,
        'opacity_scale': args.pre_opacity_scale,
        'size_shrink': args.pre_size_shrink,
        'min_views': args.pre_min_consistency_views,
        # [同步修改] 传递高级参数，对应 gaussian_model.py 中的新接口
        'dt_quantile': args.pre_dt_quantile,
        'len_threshold_mult': args.pre_len_threshold_mult,
        'align_ql': args.pre_align_ql,
        'align_qh': args.pre_align_qh,
        'eta_o_quantile': args.pre_eta_o_quantile,
        'ratio_clamp_min': args.pre_ratio_clamp_min,
        'ratio_clamp_max': args.pre_ratio_clamp_max,
        'visible_count_threshold': args.pre_visible_count_threshold
    }

    # 执行训练并获取最终指标
    final_res = training(lp.extract(args), op.extract(args), pp.extract(args),
                         args.test_iterations, args.save_iterations, args.checkpoint_iterations,
                         args.start_checkpoint, args.debug_from, pre_densify_args)

    # ==========================================================
    # --- 最终结果看板 ---
    # ==========================================================
    scene_name = os.path.basename(os.path.abspath(args.source_path))
    print("\n" + "=" * 60)
    print(f" 场景 [{scene_name}] 训练与评估完成！")
    print("-" * 60)
    print(f" 最终平均指标汇总 (测试集):")
    print(f"  >> PSNR :  {final_res['psnr']:.4f}")
    print(f"  >> SSIM :  {final_res['ssim']:.4f}")
    print(f"  >> LPIPS:  {final_res['lpips']:.4f}")
    print("=" * 60)

    # 将结果持久化保存到文件
    res_file = os.path.join(args.model_path, "final_metrics.txt")
    with open(res_file, "w") as f:
        f.write(f"PSNR: {final_res['psnr']:.4f}\n")
        f.write(f"SSIM: {final_res['ssim']:.4f}\n")
        f.write(f"LPIPS: {final_res['lpips']:.4f}\n")

    print(f"结果已记录至: {res_file}\n")