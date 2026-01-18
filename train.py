# 版权声明（保留原版权信息）
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr

import os
import torch
import torchvision
from random import randint
# 损失函数工具：L1损失、结构相似性（SSIM）损失
from utils.loss_utils import l1_loss, ssim
# 3D高斯渲染核心模块、网络GUI可视化
from gaussian_renderer import render, network_gui
import sys
# 场景管理、高斯模型核心类
from scene import Scene, GaussianModel
# 通用工具：随机数种子固定等
from utils.general_utils import safe_state
import uuid
# 进度条可视化
from tqdm import tqdm
# 图像评估指标：峰值信噪比（PSNR）
from utils.image_utils import psnr
# 命令行参数解析
from argparse import ArgumentParser, Namespace
# 自定义参数类：模型参数、渲染管线参数、优化参数
from arguments import ModelParams, PipelineParams, OptimizationParams
# 图像保存工具
from torchvision.utils import save_image
from torch import nn
import copy

# 尝试导入Tensorboard（训练日志可视化）
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             pre_densify_args):
    """
    核心训练函数：实现3D高斯溅射（3DGS）的完整训练流程
    参数说明：
        dataset: 数据集参数（路径、分辨率、背景等）
        opt: 优化器参数（学习率、迭代次数、密集化阈值等）
        pipe: 渲染管线参数（渲染分辨率、是否使用白色背景等）
        testing_iterations: 测试评估的迭代次数列表（如[5000, 10000]）
        saving_iterations: 模型保存的迭代次数列表
        checkpoint_iterations: 检查点保存的迭代次数列表
        checkpoint: 预加载的检查点路径（断点续训用）
        debug_from: 从哪个迭代开始开启调试模式
        pre_densify_args: 高斯预密集化策略参数（字典形式）
    """
    # 初始迭代次数（断点续训时会覆盖）
    first_iter = 0
    # 初始化输出文件夹和Tensorboard日志器
    tb_writer = prepare_output_and_logger(dataset)
    # 初始化3D高斯模型（传入球谐函数阶数）
    gaussians = GaussianModel(dataset.sh_degree)
    # 初始化场景（加载相机、点云、图像等数据）
    scene = Scene(dataset, gaussians)
    # 为高斯模型设置优化器（AdamW）和学习率调度
    gaussians.training_setup(opt)

    # 断点续训：加载预训练的检查点
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    # 设置背景颜色：白色背景/黑色背景
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    # ---------------------------------高斯预密集化策略（新增）---------------------------------
    # 将旧的逻辑改为调用 apply_SaGPD
    if pre_densify_args is not None and pre_densify_args['enabled']:
        try:
            # 调用 SaGPD 模块
            gaussians.apply_SaGPD(
                scene_extent=scene.cameras_extent,
                K=8,  # KNN 邻居数 [cite: 71]
                tau_s_quantile=0.7,  # 稀疏度分位数阈值 [cite: 75]
                gamma_o=0.4,  # 不透明度因子 [cite: 92]
                delta=1.5  # 尺度缩小系数 [cite: 96]
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"SaGPD 失败：显存不足 - {e}")
            torch.cuda.empty_cache()


    # CUDA事件：用于统计单次迭代的耗时
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # 测试图像保存目录（存储各迭代的渲染结果和GT）
    test_imgs_dir = os.path.join(args.model_path, "test_imgs/")
    os.makedirs(test_imgs_dir, exist_ok=True)

    # 训练视角缓存栈（随机抽取训练相机视角）
    viewpoint_stack = None
    # 损失指数移动平均（用于日志可视化，平滑损失曲线）
    ema_loss_for_log = 0.0
    # 训练进度条（可视化迭代进度）
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="训练进度")
    first_iter += 1
    # 背景掩码（未使用，预留）
    bg_mask = None
    # 损失累加器（用于密集化阶段的损失统计）
    loss_accum = 0
    # 伪标签栈（未使用，预留）
    pseudo_stack = None

    # 核心训练循环
    for iteration in range(first_iter, opt.iterations + 1):
        # 记录迭代开始时间
        iter_start.record()

        # 更新高斯模型的学习率（按预设的学习率调度策略）
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代提升球谐函数（SH）阶数（提升颜色表达能力）
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机抽取一个训练相机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()  # 重新填充视角栈
        # 随机弹出一个视角（无放回抽样）
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        # 加载该视角的真实图像（GT）并移到CUDA
        gt_image = viewpoint_cam.original_image.cuda()

        # 开启调试模式（渲染时输出更多中间结果）
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 渲染背景：随机背景（训练增强）/固定背景
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        # 执行3D高斯渲染：得到渲染图像、视角空间点、可见性掩码、高斯半径
        render_pkg = render(
            viewpoint_cam,  # 当前渲染相机
            gaussians,  # 高斯模型
            pipe,  # 渲染管线参数
            bg,  # 背景颜色
            is_train=True,  # 训练模式（会输出更多中间张量）
            iteration=iteration  # 当前迭代次数
        )


        # 解析渲染结果
        image = render_pkg["render"]  # 渲染图像（预测结果）
        viewspace_point_tensor = render_pkg["viewspace_points"]  # 视角空间下的高斯点坐标
        visibility_filter = render_pkg["visibility_filter"]  # 可见高斯点掩码（是否被相机看到）
        radii = render_pkg["radii"]  # 每个高斯在图像平面的投影半径

        # 计算损失：L1损失（像素级误差） + SSIM损失（结构相似性）
        Ll1 = l1_loss(image, gt_image)  # L1损失：|预测-真实|的均值
        ssim_value = ssim(image, gt_image)  # SSIM值：越接近1，结构越相似
        # 总损失 = L1损失 + 权重*(1-SSIM)（将SSIM转化为损失，越小越好）
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # 反向传播：计算梯度
        loss.backward()
        # 记录迭代结束时间
        iter_end.record()

        # 无梯度上下文（仅用于日志、保存、密集化等操作，避免计算梯度）
        with torch.no_grad():
            # 进度条更新：仅在密集化阶段累加损失
            if iteration > opt.densify_from_iter:
                loss_accum += loss.clone().detach().item()

            # 更新损失的指数移动平均（平滑损失，避免波动）
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # 每10次迭代更新进度条（减少IO开销）
            if iteration % 10 == 0:
                progress_bar.set_postfix({"损失值": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            # 训练结束时关闭进度条
            if iteration == opt.iterations:
                progress_bar.close()

            # 日志记录与模型保存
            training_report(
                dataset, tb_writer, iteration, loss, l1_loss,
                iter_start.elapsed_time(iter_end), testing_iterations,
                scene, render, (pipe, background)
            )
            # 保存模型（按指定迭代次数）
            if (iteration in saving_iterations):
                print(f"\n[迭代 {iteration}] 保存高斯模型")
                scene.save(iteration)

            # 高斯密集化与剪枝（核心：动态调整高斯数量）
            if iteration < opt.densify_until_iter:
                # 更新高斯的最大投影半径（用于判断是否需要密集化）
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                # 统计密集化指标（梯度、投影大小等）
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 按间隔执行密集化+剪枝
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    # 密集化：拆分高梯度高斯；剪枝：移除低贡献/小尺寸高斯
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,  # 密集化梯度阈值（高梯度高斯才拆分）
                        0.005,  # 剪枝阈值（低权重高斯被移除）
                        scene.cameras_extent,  # 场景范围（限制高斯位置）
                        size_threshold  # 尺寸阈值（未使用）
                    )

                # 按间隔重置不透明度（避免不透明度饱和）
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步进（更新高斯参数：位置、尺度、不透明度、颜色等）
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                # 清空梯度（避免梯度累积）
                gaussians.optimizer.zero_grad(set_to_none=True)

            # 保存检查点（断点续训用，包含模型参数+当前迭代次数）
            if (iteration in checkpoint_iterations):
                print(f"\n[迭代 {iteration}] 保存检查点")
                torch.save(
                    (gaussians.capture(), iteration),  # capture()：保存高斯的所有可训练参数
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                )


def prepare_output_and_logger(args):
    """
    初始化训练输出文件夹和Tensorboard日志器
    参数：
        args: 数据集/模型参数
    返回：
        tb_writer: Tensorboard日志器（None表示未安装）
    """
    # 若未指定模型输出路径，生成随机唯一ID作为路径
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # 创建输出文件夹（不存在则创建）
    print(f"训练输出文件夹：{args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    # 保存训练配置参数（方便复现）
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 初始化Tensorboard日志器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("未安装Tensorboard：跳过训练进度日志记录")
    return tb_writer


def training_report(args, tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    """
    训练日志报告：记录损失、迭代时间，在指定迭代评估测试集并保存渲染结果
    参数：
        args: 全局参数
        tb_writer: Tensorboard日志器
        iteration: 当前迭代次数
        loss: 本次迭代总损失
        l1_loss: L1损失函数
        elapsed: 本次迭代耗时（ms）
        testing_iterations: 测试评估的迭代列表
        scene: 场景对象
        renderFunc: 渲染函数（gaussian_renderer.render）
        renderArgs: 渲染函数参数（管线参数、背景）
    """
    # 记录训练损失和迭代时间到Tensorboard
    if tb_writer:
        tb_writer.add_scalar('训练损失/总损失', loss.item(), iteration)
        tb_writer.add_scalar('迭代耗时/单次迭代(ms)', elapsed, iteration)

    # 在指定迭代执行测试集评估
    if iteration in testing_iterations:
        # 清空CUDA缓存（避免评估时显存不足）
        torch.cuda.empty_cache()
        # 评估配置：测试集 + 训练集抽样
        validation_configs = (
            {'name': '测试集', 'cameras': scene.getTestCameras()},
            {'name': '训练集', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(len(scene.getTrainCameras()))]}
        )

        # 遍历测试/训练集，渲染并计算评估指标
        for config in validation_configs:
            # 渲染结果保存路径
            render_path = os.path.join(args.model_path, config['name'], f"ours_{iteration}", "渲染结果")
            # 真实图像保存路径
            gts_path = os.path.join(args.model_path, config['name'], f"ours_{iteration}", "真实图像")
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(gts_path, exist_ok=True)

            # 若有可评估的相机视角
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0  # 测试集L1损失累加
                psnr_test = 0.0  # 测试集PSNR累加

                # 遍历每个相机视角
                for idx, viewpoint in enumerate(config['cameras']):
                    # 渲染该视角的图像
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg["render"]
                    # 加载真实图像并限制范围在[0,1]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    # 前5个视角保存到Tensorboard
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(f"{config['name']}_视角_{viewpoint.image_name}/渲染图", image[None],
                                             global_step=iteration)
                        # 首次测试时保存真实图像
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"{config['name']}_视角_{viewpoint.image_name}/真实图", gt_image[None],
                                                 global_step=iteration)

                    # 累加L1损失和PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    # 保存渲染结果和真实图像
                    save_image(image, os.path.join(render_path, f'{idx:05d}.png'))
                    save_image(gt_image, os.path.join(gts_path, f'{idx:05d}.png'))

                # 计算平均L1和PSNR
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[迭代 {iteration}] 评估{config['name']}：L1损失 {l1_test:.4f} | PSNR {psnr_test:.4f}")

                # 记录到Tensorboard
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/L1损失", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/PSNR", psnr_test, iteration)

        # 记录高斯不透明度直方图和总数量到Tensorboard
        if tb_writer:
            tb_writer.add_histogram("高斯/不透明度分布", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('高斯/总数量', scene.gaussians.get_xyz.shape[0], iteration)

        # 清空CUDA缓存
        torch.cuda.empty_cache()


if __name__ == "__main__":
    """
    主函数：解析命令行参数，初始化训练环境，启动训练流程
    """
    # 创建命令行参数解析器
    parser = ArgumentParser(description="3D高斯溅射（3DGS）训练脚本参数")
    # 注册自定义参数类：模型参数（数据集路径、SH阶数等）
    lp = ModelParams(parser)
    # 优化参数（学习率、迭代次数、密集化参数等）
    op = OptimizationParams(parser)
    # 渲染管线参数（渲染分辨率、是否使用白色背景等）
    pp = PipelineParams(parser)

    # 额外自定义参数
    parser.add_argument('--ip', type=str, default="127.0.0.1", help="GUI服务器IP")
    parser.add_argument('--port', type=int, default=6009, help="GUI服务器端口")
    parser.add_argument('--debug_from', type=int, default=-1, help="从哪个迭代开始调试模式")
    parser.add_argument('--detect_anomaly', action='store_true', default=False, help="开启PyTorch梯度异常检测")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 10000], help="执行测试评估的迭代次数")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000], help="保存模型的迭代次数")
    parser.add_argument("--quiet", action="store_true", help="静默模式：关闭大部分打印输出")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[], help="保存检查点的迭代次数")
    parser.add_argument("--start_checkpoint", type=str, default=None, help="断点续训的检查点路径")

    # --- 新增自定义参数（师兄命令） ---
    parser.add_argument("--pre_densify", action="store_true", help="开启高斯预密集化（训练前拆分高斯）")
    parser.add_argument("--pre_gamma", type=float, default=2.0, help="预密集化倍率因子（控制拆分数量）")

    # 解析命令行参数
    args = parser.parse_args(sys.argv[1:])
    # 将最终迭代次数加入保存列表（训练结束时保存模型）
    args.save_iterations.append(args.iterations)

    print(f"开始训练，模型输出路径：{args.model_path}")

    # 初始化系统随机数种子（保证实验可复现）
    safe_state(args.quiet)

    # --- 关键修改：补全数据集参数 ---
    dataset_args = lp.extract(args)

    # 构造预密集化参数字典
    pre_densify_args = {
        'enabled': args.pre_densify,  # 是否开启预密集化
        'gamma': args.pre_gamma  # 密集化倍率
    }

    # 启动GUI服务器（用于可视化训练过程）
    network_gui.init(args.ip, args.port)
    # 开启梯度异常检测（调试用）
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # --- 启动核心训练流程 ---
    training(
        dataset_args,  # 数据集参数
        op.extract(args),  # 优化参数
        pp.extract(args),  # 渲染管线参数
        args.test_iterations,  # 测试迭代列表
        args.save_iterations,  # 保存迭代列表
        args.checkpoint_iterations,  # 检查点迭代列表
        args.start_checkpoint,  # 预加载检查点
        args.debug_from,  # 调试起始迭代
        pre_densify_args  # 预密集化参数
    )

    # 训练完成
    print("\n训练结束！模型已保存至：{}".format(args.model_path))