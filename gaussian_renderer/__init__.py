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

# 导入核心依赖库
import torch  # PyTorch核心库，用于张量计算和自动求导
import math  # 数学运算库，用于视场角正切值计算
# 导入高斯光栅化核心模块（3DGS的核心CUDA加速光栅化器）
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# 导入高斯模型类（存储3D高斯的位置、尺度、旋转、颜色等参数）
from scene.gaussian_model import GaussianModel
# 导入球谐函数(SH)工具函数（用于将SH特征转换为RGB颜色）
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, \
           override_color=None, is_train=False, iteration=None):
    """
    3D高斯溅射（3DGS）核心渲染函数：将3D高斯点投影到指定相机视角，生成2D渲染图像
    参数说明：
        viewpoint_camera: 相机视角对象（包含分辨率、视场角、内外参、相机中心等）
        pc (GaussianModel): 3D高斯模型实例（存储所有高斯点的位置、尺度、旋转、颜色、不透明度等参数）
        pipe: 渲染管线配置对象（包含compute_cov3D_python、convert_SHs_python等开关）
        bg_color (torch.Tensor): 背景色张量（必须在GPU上，形状[3]，值范围0-1）
        scaling_modifier: 高斯尺度修正因子（全局缩放高斯大小，默认1.0）
        override_color: 覆盖高斯颜色（若不为None，直接使用该颜色渲染，跳过SH计算）
        is_train: 是否为训练模式（训练模式启用DropGaussian增强）
        iteration: 当前训练迭代数（仅用于DropGaussian的丢弃率计算）
    返回值：
        dict: 包含渲染图像、2D屏幕坐标、可见高斯掩码、高斯投影半径的字典
    """

    # ===================== 1. 初始化2D屏幕坐标张量（用于梯度追踪） =====================
    # 创建与高斯3D位置同形状的零张量，开启梯度追踪（requires_grad=True）
    # 作用：反向传播时，追踪3D高斯位置变化对2D屏幕坐标的影响，从而优化3D位置
    screenspace_points = torch.zeros_like(
        pc.get_xyz,  # 参考高斯3D位置张量的形状和设备
        dtype=pc.get_xyz.dtype,  # 保持数据类型一致
        requires_grad=True,  # 开启自动求导
        device="cuda"  # 强制在GPU上（渲染全程在GPU执行）
    ) + 0  # +0是为了避免张量共享内存（可选，增强鲁棒性）
    try:
        # 强制保留该张量的梯度（即使该张量不是loss直接依赖，也保留梯度）
        # 解决某些场景下梯度被自动释放的问题
        screenspace_points.retain_grad()
    except:
        # 兼容低版本PyTorch或特殊设备，保留梯度失败则跳过
        pass

    # ===================== 2. 配置光栅化核心参数 =====================
    # 计算相机视场角的正切值（光栅化需要的核心参数，将3D坐标转换为2D屏幕坐标）
    # FoVx/FoVy: 相机水平/垂直视场角（弧度制），*0.5是因为取半角
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 构建光栅化配置对象（告诉光栅化器如何投影3D高斯到2D图像）
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 渲染图像高度（像素）
        image_width=int(viewpoint_camera.image_width),  # 渲染图像宽度（像素）
        tanfovx=tanfovx,  # 水平视场角半角正切值
        tanfovy=tanfovy,  # 垂直视场角半角正切值
        bg=bg_color,  # 背景色（GPU张量）
        scale_modifier=scaling_modifier,  # 高斯尺度全局修正因子
        viewmatrix=viewpoint_camera.world_view_transform,  # 视图矩阵（将世界坐标转相机坐标）
        projmatrix=viewpoint_camera.full_proj_transform,  # 投影矩阵（将相机坐标转裁剪坐标）
        sh_degree=pc.active_sh_degree,  # 球谐函数阶数（控制颜色细节）
        campos=viewpoint_camera.camera_center,  # 相机中心坐标（世界坐标系）
        prefiltered=False,  # 是否预过滤高斯（默认关闭）
        debug=pipe.debug,  # 是否开启调试模式（输出中间数据）
    )

    # 创建高斯光栅化器实例（核心渲染器，基于CUDA加速）
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ===================== 3. 提取高斯核心参数 =====================
    means3D = pc.get_xyz  # 所有高斯的3D位置（形状[N,3]，N为高斯数量）
    means2D = screenspace_points  # 初始化的2D屏幕坐标（后续由光栅化器更新）
    opacity = pc.get_opacity  # 所有高斯的不透明度（形状[N,1]，值范围0-1）

    # 3.1 协方差参数准备（高斯的形状描述，可选预计算或光栅化器计算）
    scales = None  # 高斯尺度（形状[N,3]）
    rotations = None  # 高斯旋转（四元数，形状[N,4]）
    cov3D_precomp = None  # 预计算的3D协方差矩阵（形状[N,6]）
    if pipe.compute_cov3D_python:
        # 若开启Python端预计算协方差：在Python中计算后传给光栅化器
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # 否则：将尺度和旋转传给光栅化器，由CUDA端快速计算协方差（效率更高）
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 3.2 颜色参数准备（可选SH转换、预计算颜色或覆盖颜色）
    shs = None  # 球谐函数特征（形状[N, 3, (SH阶数+1)^2]）
    colors_precomp = None  # 预计算的RGB颜色（形状[N,3]）
    if override_color is None:
        # 未指定覆盖颜色：计算高斯的RGB颜色
        if pipe.convert_SHs_python:
            # 开启Python端SH转RGB：在Python中计算颜色后传给光栅化器
            # 重塑SH特征形状，适配eval_sh函数输入
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            # 计算高斯到相机的方向向量（用于SH颜色计算）
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)  # 归一化方向向量
            # SH特征转换为RGB颜色（值范围-0.5~0.5）
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 颜色偏移并裁剪（转换为0~1范围）
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 否则：将SH特征传给光栅化器，由CUDA端快速转换为RGB（效率更高）
            shs = pc.get_features
    else:
        # 指定了覆盖颜色：直接使用该颜色（用于调试或特殊渲染需求）
        colors_precomp = override_color

    # ===================== 4. DropGaussian增强（仅训练模式） =====================
    # 作用：随机降低部分高斯的不透明度，避免模型过度依赖某几个高斯点，提升泛化能力
    if is_train:
        # 创建初始补偿因子（全1张量，形状[N]，每个高斯对应一个因子）
        compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
        # 计算丢弃率：随迭代数线性增加（0~0.2），迭代10000轮时丢弃率达20%
        drop_rate = 0.2 * (iteration / 10000)
        # 创建Dropout层（按概率将补偿因子置0）
        d = torch.nn.Dropout(p=drop_rate)
        # 应用Dropout：部分补偿因子变为0
        compensation = d(compensation)
        # 将补偿因子应用到不透明度（[:, None]是为了维度匹配：[N]→[N,1]）
        opacity = opacity * compensation[:, None]

    # ===================== 5. 核心光栅化渲染 =====================
    # 调用CUDA光栅化器，将3D高斯投影为2D图像
    # 输入：高斯3D位置、初始2D坐标、SH特征/预计算颜色、不透明度、尺度/旋转/协方差
    # 输出：渲染图像（形状[H,W,3]）、每个高斯的2D投影半径（形状[N]）
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # ===================== 6. 结果处理与返回 =====================
    # 裁剪渲染图像像素值到0~1范围（避免数值溢出）
    rendered_image = rendered_image.clamp(0, 1)
    # 构建输出字典：包含训练/后续处理需要的核心数据
    out = {
        "render": rendered_image,  # 最终2D渲染图像（[H,W,3]）
        "viewspace_points": screenspace_points,  # 带梯度的2D屏幕坐标（[N,3]）
        "visibility_filter": (radii > 0).nonzero(),  # 可见高斯掩码（[M,1]，M为可见高斯数）
        "radii": radii  # 每个高斯的2D投影半径（[N]）
    }

    return out