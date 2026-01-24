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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2  # CUDA加速的KNN距离计算
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math
import cv2
from tqdm import tqdm
# 必须确保 diff-gaussian-rasterization 已安装，用于 Fallback 模式
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class GaussianModel:
    """
    3D高斯溅射模型核心类，负责管理所有高斯点的参数（位置、颜色、尺度、旋转、不透明度等）
    以及高斯点的密集化、剪枝、保存/加载等核心操作
    """

    def setup_functions(self):
        """
        初始化各种激活函数和协方差矩阵构建函数
        这些函数用于将网络学习的原始参数转换为实际物理意义的参数
        """

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """
            从缩放和旋转参数构建协方差矩阵
            Args:
                scaling: 缩放参数 (经过exp激活)
                scaling_modifier: 缩放修正因子
                rotation: 旋转四元数 (经过normalize激活)
            Returns:
                对称化的协方差矩阵
            """
            # 构建缩放-旋转矩阵
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # 计算协方差矩阵: L * L^T
            actual_covariance = L @ L.transpose(1, 2)
            # 提取对称矩阵的唯一元素（去除冗余）
            symm = strip_symmetric(actual_covariance)
            return symm

        # 缩放参数的激活函数：exp确保缩放值为正
        self.scaling_activation = torch.exp
        # 缩放参数的逆激活函数：用于初始化
        self.scaling_inverse_activation = torch.log

        # 协方差矩阵构建函数
        self.covariance_activation = build_covariance_from_scaling_rotation

        # 不透明度激活函数：sigmoid确保值在0-1之间
        self.opacity_activation = torch.sigmoid
        # 不透明度逆激活函数：用于初始化
        self.inverse_opacity_activation = inverse_sigmoid

        # 旋转参数激活函数：归一化四元数
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        """
        初始化高斯模型
        Args:
            sh_degree: Spherical Harmonics（球谐函数）的最大阶数
        """
        # 当前激活的球谐函数阶数（训练中逐步增加）
        self.active_sh_degree = 0
        # 最大球谐函数阶数
        self.max_sh_degree = sh_degree

        # 高斯点核心参数（均为未激活的原始值，使用nn.Parameter包装以支持梯度更新）
        self._xyz = torch.empty(0)  # 3D位置 [N, 3]
        self._features_dc = torch.empty(0)  # 球谐函数DC分量 [N, 1, 3] (基础颜色)
        self._features_rest = torch.empty(0)  # 球谐函数高阶分量 [N, (SH_degree^2-1), 3] (细节颜色)
        self._scaling = torch.empty(0)  # 缩放参数（原始值）[N, 3]
        self._rotation = torch.empty(0)  # 旋转四元数（原始值）[N, 4]
        self._opacity = torch.empty(0)  # 不透明度参数（原始值）[N, 1]

        # 2D屏幕空间最大半径（用于可见性剪枝）
        self.max_radii2D = torch.empty(0)
        # 位置梯度累积器（用于密集化判断）
        self.xyz_gradient_accum = torch.empty(0)
        # 梯度累积计数分母
        self.denom = torch.empty(0)
        # 优化器
        self.optimizer = None
        # 密集化阈值参数
        self.percent_dense = 0
        # 空间学习率缩放因子
        self.spatial_lr_scale = 0

        # 初始化激活函数
        self.setup_functions()

    def capture(self):
        """
        捕获模型当前状态，用于保存检查点
        Returns:
            包含所有模型参数和优化器状态的元组
        """
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        """
        从检查点恢复模型状态
        Args:
            model_args: 从capture()保存的模型参数元组
            training_args: 训练参数
        """
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args

        # 重新设置训练相关参数
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # 恢复优化器状态
        self.optimizer.load_state_dict(opt_dict)

    # -------------------------- 属性访问器（激活后的值） --------------------------
    @property
    def get_scaling(self):
        """获取激活后的缩放值（经过exp）"""
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        """获取激活后的旋转四元数（经过normalize）"""
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        """获取高斯点3D位置（无需激活）"""
        return self._xyz

    @property
    def get_features(self):
        """获取完整的球谐函数特征（DC + 高阶）"""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        """获取激活后的不透明度（经过sigmoid）"""
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        """
        获取高斯点的协方差矩阵
        Args:
            scaling_modifier: 缩放修正因子（用于训练过程中的调整）
        Returns:
            每个高斯点的协方差矩阵
        """
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """提升球谐函数的激活阶数（训练中逐步启用高阶特征）"""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        从点云初始化高斯模型
        Args:
            pcd: 输入点云 (BasicPointCloud格式)
            spatial_lr_scale: 空间学习率缩放因子
        """
        self.spatial_lr_scale = spatial_lr_scale

        # 将点云转换为CUDA张量
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # RGB颜色转换为球谐函数DC分量
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # 初始化球谐函数特征：DC分量存储基础颜色，高阶分量初始化为0
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color  # DC分量
        features[:, 3:, 1:] = 0.0  # 高阶分量初始化为0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 计算每个点到最近邻的距离（用于初始化缩放参数）
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda())[0], 0.0000001)
        # 初始化缩放参数（使用log逆激活）
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # 初始化旋转参数（单位四元数，无旋转）
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 初始化不透明度参数（逆sigmoid激活，初始值0.1）
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 将所有参数包装为nn.Parameter以支持梯度更新
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        """
        设置训练相关的优化器和学习率调度
        Args:
            training_args: 训练参数配置
        """
        self.percent_dense = training_args.percent_dense
        # 初始化梯度累积器
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 定义优化器参数组（不同参数使用不同学习率）
        l = [
            # 位置参数：使用空间缩放的学习率
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # 球谐函数DC分量：基础颜色
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # 球谐函数高阶分量：细节颜色，学习率更低
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # 不透明度参数
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # 缩放参数
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # 旋转参数
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 位置参数的学习率调度器（指数衰减）
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        '''
        每步更新学习率（主要更新位置参数的学习率）
        Args:
            iteration: 当前训练迭代步数
        Returns:
            更新后的位置参数学习率
        '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        """
        构建保存PLY文件时的属性列表
        Returns:
            属性名称列表
        """
        # 基础属性：位置和法向量（法向量未使用，设为0）
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        # 球谐函数DC分量
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))

        # 球谐函数高阶分量
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))

        # 不透明度
        l.append('opacity')

        # 缩放参数
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))

        # 旋转参数
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l

    def save_ply(self, path):
        """
        将高斯模型保存为PLY文件
        Args:
            path: 保存路径
        """
        # 创建目录（如果不存在）
        mkdir_p(os.path.dirname(path))

        # 提取所有参数并转换为CPU numpy数组
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)  # 法向量设为0
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # 构建PLY文件的数据类型
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # 组合所有属性
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))

        # 写入PLY文件
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        """
        重置不透明度参数（将不透明度限制在0.01以内）
        """
        # 将不透明度限制在0.01以内，然后转换为原始值
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        # 更新优化器中的不透明度参数
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        """
        从PLY文件加载高斯模型
        Args:
            path: PLY文件路径
        """
        plydata = PlyData.read(path)

        # 读取位置
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        # 读取不透明度
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # 读取球谐函数DC分量
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # 读取球谐函数高阶分量
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # 重塑为[点数, 3, 高阶分量数]
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # 读取缩放参数
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 读取旋转参数
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 将所有参数转换为CUDA的nn.Parameter
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # 激活最大阶数的球谐函数
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        替换优化器中的指定参数
        Args:
            tensor: 新的参数张量
            name: 参数名称（对应优化器参数组的name）
        Returns:
            更新后的可优化张量字典
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # 获取原参数的优化器状态
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    # 重置动量状态
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # 删除原参数的状态
                del self.optimizer.state[group['params'][0]]
                # 替换为新参数
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                # 恢复状态（如果有）
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        根据掩码剪枝优化器中的参数
        Args:
            mask: 保留的参数掩码 (True表示保留)
        Returns:
            剪枝后的可优化张量字典
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 剪枝动量状态
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 删除原参数状态
                del self.optimizer.state[group['params'][0]]
                # 剪枝参数并重新包装
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                # 恢复状态
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 无状态时直接剪枝
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        根据掩码剪枝高斯点
        Args:
            mask: 要删除的点掩码 (True表示删除)
        """
        # 有效点掩码：取反
        valid_points_mask = ~mask
        # 剪枝优化器中的参数
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 更新所有参数
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 剪枝辅助变量
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        向优化器中追加新的参数张量
        Args:
            tensors_dict: 新参数字典 {参数名: 新张量}
        Returns:
            更新后的可优化张量字典
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # 获取要追加的张量
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                # 扩展动量状态
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                # 删除原参数状态
                del self.optimizer.state[group['params'][0]]
                # 追加新参数
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                # 恢复状态
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 无状态时直接追加
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        """
        密集化后处理：将新生成的高斯点添加到模型中
        Args:
            new_xyz: 新点位置
            new_features_dc: 新点DC特征
            new_features_rest: 新点高阶特征
            new_opacities: 新点不透明度
            new_scaling: 新点缩放
            new_rotation: 新点旋转
        """
        # 构建新参数字典
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation
        }

        # 追加到优化器
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        # 更新模型参数
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 重置辅助变量
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros(self.get_xyz.shape[0] - self.max_radii2D.shape[0]).to(self.max_radii2D)])

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        密集化策略1：分裂大的高斯点（基于梯度和尺度）
        Args:
            grads: 位置梯度
            grad_threshold: 梯度阈值
            scene_extent: 场景范围
            N: 每个点分裂出的新点数
        """
        n_init_points = self.get_xyz.shape[0]
        # 填充梯度以匹配点数
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        # 选择满足条件的点：梯度大于阈值 且 尺度足够大
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        # 为选中的点生成新的高斯点
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)  # 缩放作为标准差
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)  # 正态分布采样

        # 应用旋转
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        # 新点的缩放（更小）
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        # 继承旋转、颜色、不透明度
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # 添加新点
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # 剪枝原有的大高斯点
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        密集化策略2：克隆小的高斯点（基于梯度和尺度）
        Args:
            grads: 位置梯度
            grad_threshold: 梯度阈值
            scene_extent: 场景范围
        """
        # 选择满足条件的点：梯度大于阈值 且 尺度足够小
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        # 克隆选中的点（完全复制参数）
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # 添加新点
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        执行完整的密集化和剪枝流程
        Args:
            max_grad: 梯度阈值（用于密集化）
            min_opacity: 最小不透明度（用于剪枝）
            extent: 场景范围
            max_screen_size: 最大屏幕尺寸（用于剪枝）
        """
        # 计算平均梯度（梯度累积 / 累积次数）
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 执行两种密集化策略
        self.densify_and_clone(grads, max_grad, extent)  # 克隆小点
        self.densify_and_split(grads, max_grad, extent)  # 分裂大点

        # 构建剪枝掩码：低不透明度 或 屏幕尺寸过大 或 世界空间尺寸过大
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size  # 屏幕空间过大
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  # 世界空间过大
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # 执行剪枝
        self.prune_points(prune_mask)

        # 清理显存
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        累积位置梯度统计（用于密集化判断）
        Args:
            viewspace_point_tensor: 视图空间的点张量（带梯度）
            update_filter: 更新掩码（哪些点需要累积）
        """
        # 累积2D梯度的范数
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        # 累积计数
        self.denom[update_filter] += 1





    @torch.no_grad()  # 禁用梯度计算，预致密化无反向传播需求，节省显存并加速
    def apply_SaGPD(self, scene, pipe, background,
                    knn_neighbors=16,  # [Paper] K=16（kNN近邻数，论文固定超参）
                    sparsity_threshold=0.7,  # [Paper] τ_s=Q(D_b,0.7)（稀疏触发阈值，分位数算子）
                    perturb_strength=0.0,  # [Paper] Lite++ 移除切平面扰动（避免引入无效噪声）
                    min_views=2,  # [Paper] M=2（候选点需通过的最小视角数，硬阈值）
                    opacity_scale=0.3,  # [Paper] γ_o=0.3（新增点不透明度衰减系数）
                    size_shrink=1.5,  # [Paper] δ=1.5（新增点尺度收缩系数）
                    depth_error=0.03):  # Fallback模式固定深度误差阈值（无DPT时使用）
        """
        [Final Fix Version 2] Coverage-Adaptive DropGaussian (aGPD-Lite++)
        核心流程：DPT加载→深度对齐→稀疏触发→候选生成→多视图过滤→几何纠偏→门控添加
        适配场景：支持DPT深度约束（稳健模式）和内部渲染深度（降级模式）
        """


        print(f"\n[aGPD-Lite++] Starting Pre-densification (Dimension Fix)...")
        torch.cuda.empty_cache()  # 初始清理显存，避免溢出

        # 获取当前所有高斯点的三维坐标 (N, 3)，N为原始点数
        xyz = self.get_xyz
        # 获取训练集相机集合（多视图校验的基础）
        train_cameras = scene.getTrainCameras()
        if len(train_cameras) == 0:
            print("[Warning] No training cameras found.")
            return

        # =================================================================
        # 0. 模式检测与 DPT 加载 (Mode Detection & DPT Loading)
        # 核心逻辑：优先使用DPT预测深度（论文推荐，提供稳健约束），加载失败则切换到Fallback模式
        # =================================================================
        dpt_depth_maps = []  # 存储所有相机的归一化DPT深度图
        dpt_mode = True  # 标记是否启用DPT模式

        # 定位DPT深度图目录：依赖scene的source_path（数据集根目录）
        if hasattr(scene, "source_path"):
            dpt_dir = os.path.join(scene.source_path, "depths_dpt")  # 标准路径：数据集根目录/depths_dpt
        else:
            print("[Error] 'scene' object missing 'source_path'.")
            dpt_mode = False  # 缺少关键属性，强制切换Fallback模式
            dpt_dir = ""

        # 检查DPT目录是否存在，不存在则切换Fallback模式
        if dpt_mode and not os.path.exists(dpt_dir):
            print(f"[Warning] DPT folder not found at: {dpt_dir}")
            print(f"[Info] Switching to FALLBACK MODE (Internal Rendering).")
            dpt_mode = False

        # 加载DPT深度图（仅DPT模式）
        if dpt_mode:
            print(f"[Info] Loading DPT depths from: {dpt_dir}")
            for cam in tqdm(train_cameras, desc="Loading DPT"):
                basename = cam.image_name  # 从相机对象获取图像basename（不含后缀）
                dpt_path = os.path.join(dpt_dir, basename + ".png")  # DPT默认存储为png格式

                # 兼容jpg格式的DPT文件（若png不存在则尝试jpg）
                if not os.path.exists(dpt_path):
                    dpt_path_jpg = os.path.join(dpt_dir, basename + ".jpg")
                    if os.path.exists(dpt_path_jpg):
                        dpt_path = dpt_path_jpg
                    else:
                        print(f"[Warning] Missing DPT file: {dpt_path}")
                        dpt_mode = False  # 任一DPT文件缺失，切换Fallback模式
                        break

                # 读取DPT深度图：IMREAD_UNCHANGED保留原始深度值（支持16bit高精度）
                dpt_img = cv2.imread(dpt_path, cv2.IMREAD_UNCHANGED)
                if dpt_img is None:
                    print(f"[Warning] Failed to read image: {dpt_path}")
                    dpt_mode = False
                    break

                # 转换为Tensor并移至GPU
                dpt_tensor = torch.tensor(dpt_img.astype(np.float32), device="cuda")

                # 调整DPT尺寸至与相机图像一致（避免投影采样时尺寸不匹配）
                if dpt_tensor.shape[0] != cam.image_height or dpt_tensor.shape[1] != cam.image_width:
                    dpt_tensor = torch.nn.functional.interpolate(
                        dpt_tensor.unsqueeze(0).unsqueeze(0),  # 扩展为(1,1,H,W)适配interpolate
                        size=(cam.image_height, cam.image_width),
                        mode='bilinear', align_corners=False  # 双线性插值，平衡精度与速度
                    ).squeeze()  # 压缩回( H,W )

                # DPT深度归一化到[0,1]（DPT输出为相对深度，需统一尺度）
                d_min, d_max = dpt_tensor.min(), dpt_tensor.max()
                dpt_norm = (dpt_tensor - d_min) / (d_max - d_min + 1e-8)  # +1e-8防止除0
                dpt_depth_maps.append(dpt_norm)

        # =================================================================
        # A. 深度对齐 (Alignment) - 论文步骤A
        # 逻辑：DPT模式→分位数两点法对齐到相机真实深度；Fallback模式→渲染自身深度作为参考
        # =================================================================
        alignment_params = []  # 存储每个相机的DPT对齐参数 (a_c, b_c)，满足 z = a_c*d_DPT + b_c
        ref_depths_fallback = []  # Fallback模式下，存储内部渲染的深度图（替代DPT）

        if dpt_mode:
            print("[Step A] Aligning DPT depths...")
            # 构造高斯点齐次坐标 (N,4) = [x,y,z,1]，方便相机矩阵变换
            means3D_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)

            # 遍历每个相机，计算对齐参数
            for idx, cam in enumerate(train_cameras):
                # 1. 高斯点投影到相机裁剪空间，用于视锥剔除
                p_homo = means3D_homo @ cam.full_proj_transform  # (N,4)：世界→裁剪空间
                p_w = 1.0 / (p_homo[:, 3] + 1e-7)  # 齐次坐标归一化因子（避免w=0）
                p_proj = p_homo[:, :3] * p_w.unsqueeze(1)  # (N,3)：裁剪→NDC坐标

                # 2. 视锥剔除：仅保留相机可见的点（NDC在[-1,1]且深度>0）
                mask = (p_proj[:, 2] > 0) & (p_proj[:, 0] >= -1) & (p_proj[:, 0] <= 1) & \
                       (p_proj[:, 1] >= -1) & (p_proj[:, 1] <= 1)
                valid_idx = torch.where(mask)[0]

                # 若有效样本数<50，使用默认对齐参数（无缩放无偏移）
                if len(valid_idx) < 50:
                    alignment_params.append((1.0, 0.0))
                    continue

                # 3. 采样相机真实深度（View Space Z轴，Metric Depth）
                p_view = means3D_homo[valid_idx] @ cam.world_view_transform  # (M,4)：世界→视图空间
                z_gs = p_view[:, 2]  # (M,)：视图空间Z轴即真实深度

                # 4. 采样对应位置的DPT相对深度
                grid = p_proj[valid_idx, :2].unsqueeze(0).unsqueeze(0)  # (1,1,M,2)：适配grid_sample输入
                d_sample = torch.nn.functional.grid_sample(
                    dpt_depth_maps[idx].unsqueeze(0).unsqueeze(0),  # (1,1,H,W)：DPT深度图
                    grid, align_corners=True, padding_mode="border"  # 边界采样用边界值
                ).squeeze()  # (M,)：采样后的DPT深度

                # 5. 分位数两点法对齐（论文q_l=0.1, q_h=0.9，避免极值干扰）
                z_l, z_h = torch.quantile(z_gs, 0.1), torch.quantile(z_gs, 0.9)  # 真实深度分位数
                d_l, d_h = torch.quantile(d_sample, 0.1), torch.quantile(d_sample, 0.9)  # DPT深度分位数

                # 计算仿射参数：a_c（尺度）、b_c（偏移）
                if abs(d_h - d_l) < 1e-6:  # 避免DPT深度无变化导致除0
                    a_c = 1.0
                else:
                    a_c = (z_h - z_l) / (d_h - d_l)  # 深度尺度映射系数
                b_c = z_l - a_c * d_l  # 深度偏移系数
                alignment_params.append((a_c, b_c))
        else:
            # Fallback模式：无DPT时，用3DGS光栅化器渲染自身深度作为参考
            print("[Step A] Fallback: Rendering internal depths...")
            means3D_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)

            # 提取当前高斯点的核心属性（用于渲染）
            opac = self.get_opacity  # 不透明度 (N,1)
            scales = self.get_scaling  # 缩放因子 (N,3)
            rots = self.get_rotation  # 旋转四元数 (N,4)

            for cam in tqdm(train_cameras, desc="Rendering Internal"):
                # 配置光栅化器参数（与3DGS渲染逻辑一致）
                raster_settings = GaussianRasterizationSettings(
                    image_height=int(cam.image_height), image_width=int(cam.image_width),
                    tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
                    bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
                    viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
                    sh_degree=0, campos=cam.camera_center, prefiltered=False, debug=False
                )
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                # 计算View Space深度（作为渲染的"颜色"，用于后续校验）
                p_view = means3D_homo @ cam.world_view_transform
                depths_val = p_view[:, 2:3]  # (N,1)：每个高斯点的真实深度

                # 渲染深度图：将深度值作为RGB颜色传入（适配光栅化器接口）
                d_img, _ = rasterizer(
                    means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
                    colors_precomp=depths_val.repeat(1, 3),  # (N,3)：深度值复制3通道
                    opacities=opac, scales=scales, rotations=rots, cov3D_precomp=None
                )
                # 保留单通道深度图（[1, H, W]），常驻GPU避免数据传输开销
                ref_depths_fallback.append(d_img[0:1])


        # =================================================================
        # B & C. 候选生成 (Candidate Generation) - 论文步骤B+C合并
        # 逻辑：1. 计算邻域稀疏度，触发增密区域；2. 过长边中点插值生成候选点（每触发点最多1个）
        # =================================================================
        print("[Step B&C] Generating candidates...")
        # 计算每个高斯点的近邻平均距离（使用高效CUDA实现，对应论文D_i）
        dist2, _ = distCUDA2(xyz)  # dist2: (N,)，每个点到最近3个邻居的平均平方距离
        dist = torch.sqrt(dist2).squeeze()  # (N,)：转换为平均欧式距离

        # 归一化稀疏度 D_bi（论文核心，消除尺度影响）
        d_min, d_max = dist.min(), dist.max()
        d_bi = (dist - d_min) / (d_max - d_min + 1e-8)  # 归一化到[0,1]

        # 稀疏触发：筛选稀疏度超阈值的点（论文tau_s=Q(D_b, 0.7)）
        tau_s = torch.quantile(d_bi, sparsity_threshold)
        trigger_indices = torch.where(d_bi > tau_s)[0]  # 触发增密的点索引
        if len(trigger_indices) == 0:  # 无稀疏区域，直接返回
            return

        # 容器：存储候选点坐标、起点索引、终点索引
        candidates_mu, candidates_i, candidates_j = [], [], []
        dt = torch.quantile(dist, 0.6)  # 论文dt=Q(D, 0.6)：长边判断基准
        len_threshold = 2.0 * dt  # 过长边阈值：大于2*dt的边才生成候选点
        chunk_size = 4096  # 批处理大小，避免显存爆炸


        # 批处理触发点，生成候选点
        for k in range(0, len(trigger_indices), chunk_size):
            batch_idx = trigger_indices[k: k + chunk_size]  # 当前批次触发点索引 (B,)
            batch_xyz = xyz[batch_idx]  # 当前批次点坐标 (B,3)

            # 计算当前批次点与所有高斯点的距离，取K+1个近邻（+1是为了剔除自身）
            dists_k, idxs_k = torch.cdist(batch_xyz, xyz).topk(knn_neighbors + 1, largest=False)
            local_dists, local_idxs = dists_k[:, 1:], idxs_k[:, 1:]  # 剔除自身，保留K个近邻 (B,K)

            # 寻找每个触发点的最长边（论文选长边生成候选点，增密收益更高）
            max_d, max_local_arg = local_dists.max(dim=1)  # (B,)：每个点的最长近邻距离及索引
            valid = max_d > len_threshold  # 筛选过长边（仅长边生成候选点）

            if valid.any():
                # 提取有效触发点的全局索引和对应最长边的终点索引
                v_batch_idx = batch_idx[valid]  # 有效触发点的全局索引
                # 映射到全局索引：local_idxs[valid]是(B_valid,K)，max_local_arg[valid]是(B_valid,)
                v_target_global = local_idxs[valid].gather(1, max_local_arg[valid].unsqueeze(1)).squeeze()

                # 中点插值生成候选点（论文简化版：每触发点最多1个候选点）
                candidates_mu.append(0.5 * (xyz[v_batch_idx] + xyz[v_target_global]))
                candidates_i.append(v_batch_idx)  # 候选点起点索引
                candidates_j.append(v_target_global)  # 候选点终点索引

        # 若无有效候选点，直接返回
        if not candidates_mu:
            return
        # 合并所有批次的候选点数据
        cand_mu_cat = torch.cat(candidates_mu)  # (N_cand,3)：所有候选点坐标
        cand_i_cat = torch.cat(candidates_i)  # (N_cand,)：每个候选点的起点索引
        cand_j_cat = torch.cat(candidates_j)  # (N_cand,)：每个候选点的终点索引
        N_cand = cand_mu_cat.shape[0]
        print(f" > Generated {N_cand} candidates.")


        # =================================================================
        # D. 一致性过滤 (Consistency Filtering) - 论文步骤D
        # 核心逻辑：在候选点两端点的可见交集上，校验多视图深度一致性，统计通过视角数
        # 关键修复：Fallback模式下grid_sampler维度不匹配问题
        # =================================================================
        print(f"[Step D] Consistency Filtering...")
        pass_counts = torch.zeros(N_cand, device="cuda")  # 每个候选点通过的视角数
        accum_error = torch.zeros(N_cand, device="cuda")  # 累计深度误差（用于计算可信度）
        # 记录最佳视角信息（误差最小的视角，为后续几何纠偏做准备）
        best_view_idx = torch.full((N_cand,), -1, dtype=torch.long, device="cuda")  # 最佳视角索引
        best_z_ref = torch.zeros(N_cand, device="cuda")  # 最佳视角的参考深度
        min_view_error = torch.full((N_cand,), 1e9, device="cuda")  # 最小深度误差

        # 构造候选点齐次坐标 (N_cand,4)，方便相机矩阵变换
        cand_homo = torch.cat([cand_mu_cat, torch.ones(N_cand, 1, device="cuda")], dim=-1)

        # 遍历所有训练相机，校验候选点深度一致性
        for idx, cam in enumerate(train_cameras):
            # 1. 候选点投影到当前相机裁剪空间
            p_proj = cand_homo @ cam.full_proj_transform  # (N_cand,4)：世界→裁剪空间
            p_w = 1.0 / (p_proj[:, 3] + 1e-7)  # 归一化因子
            ndc = p_proj[:, :2] * p_w.unsqueeze(1)  # (N_cand,2)：NDC坐标（x,y）

            # 2. 视锥检查：仅保留相机视锥内的候选点
            in_view = (p_proj[:, 2] > 0) & (ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) & \
                      (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1)
            if not in_view.any():
                continue  # 无有效点，跳过当前相机
            valid_indices = torch.where(in_view)[0]  # 视锥内候选点的索引

            # 3. 构造采样网格（适配grid_sample输入格式：(1,1,M,2)）
            grid = ndc[valid_indices].unsqueeze(0).unsqueeze(0)

            # 4. 获取参考深度Z_ref和误差阈值Threshold
            if dpt_mode:
                # DPT模式：采样DPT深度并通过仿射参数对齐到真实深度
                d_raw = torch.nn.functional.grid_sample(
                    dpt_depth_maps[idx].unsqueeze(0).unsqueeze(0),
                    grid, align_corners=True, padding_mode="border"
                ).squeeze()  # (M,)：采样的DPT相对深度
                a_c, b_c = alignment_params[idx]
                z_ref = a_c * d_raw + b_c  # (M,)：对齐后的真实深度参考值
                # 动态误差阈值（论文逻辑：随深度增大而增大，避免近点过严、远点过松）
                threshold = torch.maximum(torch.tensor(0.05, device="cuda"), 0.2 * z_ref)
            else:
                # Fallback模式：采样内部渲染的深度图作为参考
                if not ref_depths_fallback:
                    continue
                # [关键修复] 维度适配：ref_depths_fallback[idx]原本是(1, H, W)（3D）
                # grid_sample要求输入为4D张量 (B, C, H, W)，故添加unsqueeze(0)扩展为(1, 1, H, W)
                input_depth = ref_depths_fallback[idx].unsqueeze(0)  # 维度转换：3D→4D
                z_ref = torch.nn.functional.grid_sample(
                    input_depth.to("cuda"),  # 4D输入：(1,1,H,W)
                    grid, align_corners=True, padding_mode="border"
                ).squeeze()  # (M,)：采样的渲染深度
                # 固定误差阈值（比例+偏移，兼容无DPT场景）
                threshold = z_ref * depth_error + 0.05


            # 5. 计算候选点的真实深度（View Space Z轴）
            p_view = cand_homo[valid_indices] @ cam.world_view_transform  # (M,4)：世界→视图空间
            z_metric = p_view[:, 2]  # (M,)：候选点的真实深度

            # 6. 深度一致性校验：误差小于阈值则通过
            diff = torch.abs(z_metric - z_ref)  # (M,)：深度绝对误差
            passed = diff < threshold  # (M,)：通过校验的掩码

            # 7. 更新通过计数和累计误差
            passed_global_idx = valid_indices[passed]
            pass_counts[passed_global_idx] += 1  # 通过视角数+1
            accum_error[passed_global_idx] += diff[passed]  # 累计误差叠加

            # 8. 更新最佳视角信息（仅保留误差最小的视角）
            current_errs = diff[passed]
            current_z_ref = z_ref[passed]
            # 筛选当前误差更小的点
            better = current_errs < min_view_error[passed_global_idx]
            update_idx = passed_global_idx[better]
            # 更新最佳视角数据
            min_view_error[update_idx] = current_errs[better]
            best_view_idx[update_idx] = idx
            best_z_ref[update_idx] = current_z_ref[better]

        # =================================================================
        # E & F. 几何纠偏 (Refine) & 门控添加 (Gated Add) - 论文步骤E+F
        # 核心逻辑：1. 几何纠偏（拉回真实表面）；2. 可信度门控初始化（抑制低可信度点干扰）
        # =================================================================
        # 最终筛选条件：1. 通过至少M个视角；2. 存在有效最佳视角（用于纠偏）
        final_mask = pass_counts >= min_views
        final_idx = torch.where(final_mask & (best_view_idx != -1))[0]

        if len(final_idx) == 0:
            print(" > No candidates passed.")
            return

        # 提取最终有效候选点的关联数据
        idx_i_final = cand_i_cat[final_idx]  # 起点索引
        idx_j_final = cand_j_cat[final_idx]  # 终点索引
        final_xyz_raw = cand_mu_cat[final_idx]  # 纠偏前的候选点坐标

        # --- [Geometric Refine: 几何纠偏（论文核心改进）] ---
        if dpt_mode:
            # 原理：沿着"相机中心→候选点"的射线，调整点的位置，使其深度等于最佳视角的参考深度
            # 解决问题：过滤仅能剔除明显错误点，无法保证点在真实表面（漂浮点）

            # 1. 收集所有相机中心和World→View变换矩阵
            all_cam_centers = torch.stack([c.camera_center for c in train_cameras])  # (C,3)
            all_w2c = torch.stack([c.world_view_transform for c in train_cameras])  # (C,4,4)

            # 2. 提取每个有效点对应的最佳相机数据
            best_cams_idx = best_view_idx[final_idx]  # (N_pts,)：最佳相机索引
            centers = all_cam_centers[best_cams_idx]  # (N_pts,3)：最佳相机中心
            w2c_mats = all_w2c[best_cams_idx]  # (N_pts,4,4)：最佳相机的变换矩阵

            # 3. 计算候选点在最佳视角下的当前深度（View Space Z）
            pts_homo = torch.cat([final_xyz_raw, torch.ones_like(final_xyz_raw[:, :1])], dim=-1)  # (N_pts,4)
            # 批量矩阵乘法：(N_pts,1,4) @ (N_pts,4,4) → (N_pts,1,4) → 压缩为(N_pts,4)
            p_view = torch.bmm(pts_homo.unsqueeze(1), w2c_mats).squeeze(1)
            current_z = p_view[:, 2]  # (N_pts,)：当前深度

            # 4. 计算射线缩放比例：target_z / current_z（将点沿射线移动到目标深度）
            target_z = best_z_ref[final_idx]  # (N_pts,)：目标深度（最佳视角的参考深度）
            ratio = target_z / (current_z + 1e-8)  # (N_pts,)：缩放比例
            ratio = torch.clamp(ratio, 0.5, 2.0)  # 限制缩放范围，避免数值爆炸

            # 5. 执行几何纠偏：NewPos = 相机中心 + 射线方向 * 缩放比例
            rays = final_xyz_raw - centers  # (N_pts,3)：相机中心到候选点的射线方向
            final_xyz = centers + rays * ratio.unsqueeze(1)  # (N_pts,3)：纠偏后的最终坐标
        else:
            # Fallback模式不进行几何纠偏（无DPT深度约束，纠偏不可靠）
            final_xyz = final_xyz_raw

        # --- [Gated Initialization: 可信度门控初始化（论文核心改进）] ---
        # 1. 继承基础属性：特征为起点和终点的平均，旋转继承起点（保证外观和朝向一致性）
        f_dc = 0.5 * (self._features_dc[idx_i_final] + self._features_dc[idx_j_final])  # 球谐DC分量 (N_pts,1,3)
        f_rest = 0.5 * (self._features_rest[idx_i_final] + self._features_rest[idx_j_final])  # 球谐Rest分量 (N_pts,15,3)
        rot = self._rotation[idx_i_final]  # 旋转四元数 (N_pts,4)

        if dpt_mode:
            # 2. 计算可信度q_new（论文公式：q_new = 通过率 * 误差抑制项）
            S_est = max(1, len(train_cameras) // 2)  # 估计的有效视角数（避免分母过小）
            term1 = torch.clamp(pass_counts[final_idx] / S_est, 0, 1)  # 通过率（0~1）
            avg_err = accum_error[final_idx] / (pass_counts[final_idx] + 1e-8)  # 平均深度误差
            # 误差抑制项（误差越小，term2越接近1）
            q_new = (term1 * torch.clamp(1.0 - avg_err / 0.5, min=0.0)).unsqueeze(1)  # (N_pts,1)

            # 3. 门控初始化不透明度和尺度（低可信度点参数更保守）
            opac = self.get_opacity[idx_i_final]  # 起点不透明度 (N_pts,1)
            scale = self.get_scaling[idx_i_final]  # 起点缩放因子 (N_pts,3)
            # 不透明度：基础衰减 * 可信度
            final_opac = self.inverse_opacity_activation(opac * opacity_scale * q_new)
            # 尺度：收缩后 * (0.5+0.5*可信度)（可信度越高，尺度越接近原始）
            scale_mod = (1.0 / size_shrink) * (0.5 + 0.5 * q_new)
            final_scale = self.scaling_inverse_activation(scale * scale_mod)
        else:
            # Fallback模式：固定参数初始化（无可信度约束）
            opac = self.get_opacity[idx_i_final]
            final_opac = self.inverse_opacity_activation(opac * opacity_scale)

            scale = self._scaling[idx_i_final]
            final_scale = scale - torch.log(torch.tensor(size_shrink, device="cuda"))

        # 调用3DGS后端接口，批量添加新增高斯点到模型中
        self.densification_postfix(final_xyz, f_dc, f_rest, final_opac, final_scale, rot)
        print(f"[SaGPD] Done. Added {len(final_xyz)} points.")