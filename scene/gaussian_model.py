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



    @torch.no_grad()  # 禁用梯度计算，节省显存并加速计算（该过程无反向传播）
    def apply_SaGPD(self, scene, pipe, background,
                    knn_neighbors=16,  # [文档参数] K=16，KNN查找的邻居数量
                    sparsity_threshold=0.7,  # [文档参数] tau_s = Q(D_b, 0.7)，稀疏度分位数阈值
                    min_views=2,  # [文档参数] M=2，通过一致性检查的最小视角数
                    opacity_scale=0.3,  # [文档参数] gamma_o=0.3，不透明度缩放系数
                    size_shrink=1.5):  # [文档参数] delta=1.5，尺寸收缩系数
        """
        [最终验证实现] aGPD-Lite++ 算法
        核心功能：基于DPT深度图和可见性约束，对稀疏高斯点云进行预致密化，补充缺失的几何细节
        """

        print(f"\n[aGPD-Lite++] 开始预致密化流程（严格匹配文档版本）...")
        torch.cuda.empty_cache()  # 清理CUDA缓存，避免显存溢出

        # 1. 基础数据准备
        xyz = self.get_xyz  # 获取当前所有高斯点的3D坐标 (N_points, 3)
        train_cameras = scene.getTrainCameras()  # 获取训练集相机列表
        if len(train_cameras) == 0:  # 无训练相机时直接返回
            return

        # =================================================================
        # [Step 0] 预计算：构建每个高斯点的可见性集合 V_i
        # 文档依据: "仅用已有可见集合信息... |Vi ∩ Vj| >= 1" [Section 4.3(i)]
        # 核心逻辑: 对每个高斯点，判断其是否在每个相机的视锥内（可见），生成可见性掩码
        # =================================================================
        print("[Step 0] 构建可见性掩码（用于后续VOV交集检查）...")
        N_points = xyz.shape[0]  # 高斯点总数
        N_cams = len(train_cameras)  # 训练相机总数

        # 显存优化：用布尔矩阵存储可见性 (N_points, N_cams)，True表示点在对应相机可见
        gaussian_vis_mask = torch.zeros((N_points, N_cams), dtype=torch.bool, device="cuda")
        # 将3D坐标转换为齐次坐标 (x,y,z,1)，方便矩阵乘法投影
        means3D_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)

        # 遍历每个相机，计算所有点在该相机的可见性
        for c_idx, cam in enumerate(train_cameras):
            # 步骤1：将3D点投影到相机的齐次裁剪空间 (N_points, 4)
            p_homo = means3D_homo @ cam.full_proj_transform
            # 齐次坐标透视除法（避免除零）
            p_w = 1.0 / (p_homo[:, 3] + 1e-7)
            # 投影到NDC空间（标准化设备坐标）(N_points, 3)
            p_proj = p_homo[:, :3] * p_w.unsqueeze(1)

            # 视锥剔除：判断点是否在相机可见范围内
            # NDC空间约束：z>0（在相机前方），x/y ∈ [-1,1]（在画面内）
            mask = (p_proj[:, 2] > 0) & (p_proj[:, 0] >= -1) & (p_proj[:, 0] <= 1) & \
                   (p_proj[:, 1] >= -1) & (p_proj[:, 1] <= 1)
            gaussian_vis_mask[:, c_idx] = mask  # 存储该相机的可见性掩码

        # =================================================================
        # [Step A] DPT深度图加载与对齐 (对应Algorithm 2 Part A)
        # 核心逻辑: 加载预计算的DPT深度图，通过分位数回归将相对深度对齐到真实度量深度
        # =================================================================
        dpt_depth_maps = []  # 存储所有相机的归一化DPT深度图
        alignment_params = []  # 存储每个相机的深度对齐参数 (a_c, b_c)，用于 z = a_c*d + b_c

        # 检查DPT深度图路径（强制依赖预生成的DPT深度）
        if hasattr(scene, "source_path"):
            dpt_dir = os.path.join(scene.source_path, "depths_dpt")  # DPT深度图存储目录
        else:
            print("[Error] Scene对象缺少'source_path'属性（无法定位DPT深度图）");
            return

        if not os.path.exists(dpt_dir):
            print(f"[Error] DPT深度图目录不存在: {dpt_dir}（已移除降级策略）");
            return

        print("[Step A] 加载并对齐DPT深度图...")

        # 1. 加载所有训练相机对应的DPT深度图
        for cam in tqdm(train_cameras, desc="加载DPT深度图"):
            basename = cam.image_name  # 相机对应的图像名称
            # 优先查找png格式，不存在则找jpg
            dpt_path = os.path.join(dpt_dir, basename + ".png")
            if not os.path.exists(dpt_path):
                dpt_path = os.path.join(dpt_dir, basename + ".jpg")

            if not os.path.exists(dpt_path):
                print(f"[Error] 相机{basename}缺少DPT深度图");
                return

            # 读取DPT深度图（无压缩格式，保留原始深度值）
            dpt_img = cv2.imread(dpt_path, cv2.IMREAD_UNCHANGED)
            # 转换为CUDA张量（float32精度）
            dpt_tensor = torch.tensor(dpt_img.astype(np.float32), device="cuda")

            # 调整深度图尺寸以匹配相机分辨率（避免投影坐标不匹配）
            if dpt_tensor.shape[0] != cam.image_height or dpt_tensor.shape[1] != cam.image_width:
                dpt_tensor = torch.nn.functional.interpolate(
                    dpt_tensor.unsqueeze(0).unsqueeze(0),  # 增加batch和channel维度
                    size=(cam.image_height, cam.image_width),  # 目标尺寸
                    mode='bilinear', align_corners=False  # 双线性插值（保持深度平滑）
                ).squeeze()  # 移除多余维度

            # 将DPT深度归一化到[0,1]（相对深度，消除不同图像的深度范围差异）
            d_min, d_max = dpt_tensor.min(), dpt_tensor.max()
            dpt_norm = (dpt_tensor - d_min) / (d_max - d_min + 1e-8)
            dpt_depth_maps.append(dpt_norm)

        # 2. 计算每个相机的深度对齐参数 (a_c, b_c)（分位数回归 + 相关性检查）
        for idx, cam in enumerate(train_cameras):
            # 步骤1：筛选该相机可见的高斯点（采样S_c集合）[对应Algo 2 Line A.1]
            visible_mask = gaussian_vis_mask[:, idx]

            # 较难理解，疑惑？-------------------------------------------------
            if visible_mask.sum() < 50:  # 可见点过少，使用默认对齐参数
                alignment_params.append((1.0, 0.0))
                continue

            # 步骤2：计算可见点的真实度量深度 z_gs（相机坐标系下的z值）
            valid_xyz = xyz[visible_mask]  # 该相机可见的高斯点坐标
            # 转换到相机视图空间 (N_valid, 4)
            p_view = torch.cat([valid_xyz, torch.ones_like(valid_xyz[:, :1])], dim=-1) @ cam.world_view_transform
            # 之后要用到---------------------------------------
            z_gs = p_view[:, 2]  # 相机视图空间的z值（真实度量深度）

            # 步骤3：从DPT深度图中采样对应位置的相对深度 d_sample
            # 将3D点投影到NDC空间，用于采样深度图
            p_homo = torch.cat([valid_xyz, torch.ones_like(valid_xyz[:, :1])], dim=-1) @ cam.full_proj_transform
            p_w = 1.0 / (p_homo[:, 3] + 1e-7)
            ndc = p_homo[:, :2] * p_w.unsqueeze(1)  # NDC空间的x/y坐标 (N_valid, 2)
            grid = ndc.unsqueeze(0).unsqueeze(0)  # 适配grid_sample的输入格式 (1,1,N_valid,2)

            # 较难理解
            # 从DPT深度图中采样对应位置也就是grid中每个坐标（当前视角下的所有可见的点的NDC坐标）的深度值
            d_sample = torch.nn.functional.grid_sample(
                dpt_depth_maps[idx].unsqueeze(0).unsqueeze(0),  # 输入深度图 (1,1,H,W)
                grid, align_corners=True, padding_mode="border"  # 边界填充避免越界
            ).squeeze()  # 输出采样深度 (N_valid,)

            # 较难理解---------------------------------------------------
            # 步骤4：相关性检查 [对应Algo 2 Line 6-7]
            # 逻辑：若真实深度z_gs与相对深度d_sample负相关，则翻转d_sample
            z_mean = z_gs.mean()  # 真实深度均值
            d_mean = d_sample.mean()  # 相对深度均值
            # 计算协方差（衡量相关性，cov<0表示负相关）
            cov = ((z_gs - z_mean) * (d_sample - d_mean)).sum()

            is_neg_corr = cov < 0
            if is_neg_corr:
                d_sample = -d_sample  # 临时翻转，保证后续分位数计算的单调性

            # 步骤5：分位数计算 [对应Algo 2 Line 8] (q_l=0.1, q_h=0.9)
            q_l, q_h = 0.1, 0.9
            z_l, z_h = torch.quantile(z_gs, q_l), torch.quantile(z_gs, q_h)  # 真实深度的分位数
            d_l, d_h = torch.quantile(d_sample, q_l), torch.quantile(d_sample, q_h)  # 相对深度的分位数


            # 步骤6：计算对齐参数 a_c (斜率) 和 b_c (截距) [对应Algo 2 Line 9]
            # 较难理解，为什么可以这么处理，或许是一个改进点---------------------
            if abs(d_h - d_l) < 1e-6:  # 深度范围过小，避免除零
                a_c = 1.0
            else:
                a_c = (z_h - z_l) / (d_h - d_l)  # 斜率：将相对深度映射到真实深度
            b_c = z_l - a_c * d_l  # 截距

            # 较难理解，这里也需要仔细对照-----------------------------------
            # 修正：若原始相关性为负，需翻转斜率（因为之前临时翻转了d_sample）
            # 推导：z = a_temp*(-d) + b = (-a_temp)*d + b → a_c = -a_temp
            if is_neg_corr:
                a_c = -a_c

            # 较难理解，为什么要保存------------------------------------
            alignment_params.append((a_c, b_c))  # 保存该相机的对齐参数




        # =================================================================
        # [Step B] 稀疏触发 (对应Algorithm 2 Part B)
        # 核心逻辑: 计算每个高斯点的稀疏度，筛选出需要补充的稀疏点（触发点）
        # =================================================================
        print("[Step B] 计算点云稀疏度，筛选触发点...")

        # 理解：这里有问题，只算3个，但是邻域取16个-------------------------------
        # 计算每个点到最近邻的平方距离（distCUDA2为CUDA加速的距离计算函数）
        dist2, _ = distCUDA2(xyz)  # 1. 得到每个点的「最近三个点的平方距离的平均数」
        dist = torch.sqrt(dist2).squeeze()  # 2. 开根号→真实欧式距离；squeeze去除冗余维度

        # 较难理解，这里是不是有问题？-----------------------------
        # 将稀疏度归一化到[0,1]（D_bi），消除绝对距离的影响
        d_min, d_max = dist.min(), dist.max()
        d_bi = (dist - d_min) / (d_max - d_min + 1e-8)

        # 计算稀疏度阈值 tau_s = Q(D_b, 0.7) [对应Algo 2 Part B]
        # 取70%分位数作为阈值，大于该值的点判定为稀疏点
        tau_s = torch.quantile(d_bi, sparsity_threshold)
        # 筛选出稀疏触发点的索引
        trigger_indices = torch.where(d_bi > tau_s)[0]

        if len(trigger_indices) == 0:  # 无稀疏点，直接返回
            return

        # =================================================================
        # [Step C] 候选点生成 (对应Algorithm 2 Part C)
        # 核心逻辑: 对每个触发点，基于KNN和可见交集约束生成候选点（中点插值）
        # 约束：1. 邻居与触发点有共同可见相机 |Vi ∩ Vu| >=1；2. 边长远大于局部均值
        # =================================================================
        print("[Step C] 基于VOV约束生成候选点...")
        candidates_mu = []  # 存储候选点的3D坐标
        candidates_i = []  # 存储候选点对应的触发点索引
        candidates_j = []  # 存储候选点对应的邻居点索引

        # 较难理解，为什么是2倍的60%
        # 计算长度阈值：2倍的60%分位数距离（过滤过短的边）
        dt = torch.quantile(dist, 0.4)
        len_threshold = 2.0 * dt
        chunk_size = 4096  # 分块处理，避免显存溢出

        # 分块处理触发点（防止一次性处理过多点导致CUDA OOM）
        for k in range(0, len(trigger_indices), chunk_size):
            batch_idx = trigger_indices[k: k + chunk_size]  # 当前块的触发点索引
            batch_xyz = xyz[batch_idx]  # 当前块的触发点坐标

            # 理解：这里才是正常的实现？--------------------------------------------------------
            # KNN查找：每个触发点找knn_neighbors个最近邻（排除自身）[对应Algo 2 Line C.1]
            dists_k, idxs_k = torch.cdist(batch_xyz, xyz).topk(knn_neighbors + 1, largest=False)
            local_dists, local_idxs = dists_k[:, 1:], idxs_k[:, 1:]  # 排除自身（第0个是自己）

            # VOV检查：触发点与邻居是否有共同可见的相机 |Vi ∩ Vu| >=1 [对应Algo 2 Line C.3]
            vis_center = gaussian_vis_mask[batch_idx]  # 触发点的可见性掩码 (B, N_cams)
            vis_neighbors = gaussian_vis_mask[local_idxs]  # 邻居的可见性掩码 (B, K, N_cams)
            # 逐邻居检查是否有共同可见相机（任意相机同时可见）
            has_overlap = (vis_center.unsqueeze(1) & vis_neighbors).any(dim=-1)  # (B, K)

            # 策略：优先选择有可见交集的邻居 (B,K)
            masked_dists_vov = local_dists.clone()
            # 在第1维过滤，第1维指的是每个邻居点，无交集的邻居标记为无效（距离=-1）
            masked_dists_vov[~has_overlap] = -1.0
            #  举例：
            # masked_dists_vov = torch.tensor([[0.1, 0.2, 0.3],
            #                                  [0.4, 0.5, 0.6]])
            # has_overlap = torch.tensor([[True, False, True],
            #                             [False, True, False]])

            # 找有交集的邻居中距离最远的（最大化候选点覆盖范围）
            max_d_vov, arg_vov = masked_dists_vov.max(dim=1)

            # 较难理解，可以这么做吗？--------------------------------------------------
            # 降级策略：若所有邻居都无交集，选择原始最远邻居 [对应Algo 2 Line C.5]
            # 返回的都是一维张量，触发点：触发点到邻居点的最大距离(和该邻居点的索引)
            max_d_raw, arg_raw = local_dists.max(dim=1)
            use_raw = (max_d_vov < 0)  # 标记需要降级的触发点
            # 最终选择的邻居索引
            # torch.where(条件张量, 真值取值张量, 假值取值张量)
            final_arg = torch.where(use_raw, arg_raw, arg_vov)
            final_max_d = torch.where(use_raw, max_d_raw, max_d_vov)

            # 长度筛选：仅保留边长远于阈值的候选 [对应Algo 2 Line C.6]
            valid = final_max_d > len_threshold

            if valid.any():
                # 筛选出有效触发点索引
                v_batch_idx = batch_idx[valid]
                # 理解：找到对应邻居的全局索引，语法较难
                v_target_global = local_idxs[valid].gather(1, final_arg[valid].unsqueeze(1)).squeeze()

                # 中点插值生成候选点 [对应Algo 2 Line C.7]
                candidates_mu.append(0.5 * (xyz[v_batch_idx] + xyz[v_target_global]))
                candidates_i.append(v_batch_idx)
                candidates_j.append(v_target_global)

        if not candidates_mu:  # 无有效候选点，直接返回
            return


        # 合并所有候选点数据
        cand_mu_cat = torch.cat(candidates_mu)  # 候选点坐标 (N_cand, 3)
        cand_i_cat = torch.cat(candidates_i)  # 对应触发点索引 (N_cand,)
        cand_j_cat = torch.cat(candidates_j)  # 对应邻居点索引 (N_cand,)
        N_cand = cand_mu_cat.shape[0]
        print(f" > 生成 {N_cand} 个候选点.")



        # =================================================================
        # [Step D] 动态分位数阈值与过滤 (对应Algorithm 2 Part D)
        # 核心逻辑: 计算候选点在各相机的深度误差，基于动态阈值筛选满足一致性的候选点
        # =================================================================
        print(f"[Step D] 动态阈值计算与候选点过滤...")

        # 初始化统计变量
        pass_counts = torch.zeros(N_cand, dtype=torch.int32, device="cuda")  # 每个候选点通过误差检查的视角数
        valid_view_counts = torch.zeros(N_cand, dtype=torch.int32, device="cuda")  # 每个候选点的有效视角数 |C_check|

        # 记录每个候选点的最佳视角（用于后续几何纠偏）
        best_view_idx = torch.full((N_cand,), -1, dtype=torch.long, device="cuda")  # 最佳视角索引
        best_z_ref = torch.zeros(N_cand, device="cuda")  # 最佳视角的目标深度z_ref
        min_view_error = torch.full((N_cand,), 1e9, device="cuda")  # 最小深度误差

        # 临时误差矩阵 (N_cand, N_cams)，存储每个候选点在每个相机的深度误差（nan表示无效）
        error_matrix = torch.full((N_cand, N_cams), float('nan'), device="cuda")

        # 候选点转换为齐次坐标，方便投影计算
        cand_homo = torch.cat([cand_mu_cat, torch.ones(N_cand, 1, device="cuda")], dim=-1)

        # Phase 1: 遍历所有相机，计算候选点的深度误差 [对应Algo 2 Part D循环]
        for idx, cam in enumerate(train_cameras):
            # 步骤1：将候选点投影到相机NDC空间，判断是否在视锥内
            p_proj = cand_homo @ cam.full_proj_transform
            p_w = 1.0 / (p_proj[:, 3] + 1e-7)
            ndc = p_proj[:, :2] * p_w.unsqueeze(1)  # NDC空间x/y坐标

            # 视锥检查：候选点是否在该相机可见范围内
            in_view = (p_proj[:, 2] > 0) & (ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) & \
                      (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1)

            if not in_view.any():  # 该相机无有效候选点，跳过
                continue

            # 理解：比较关键的变量，当前视角下的所有可见的候选点的索引
            valid_indices = torch.where(in_view)[0] # 返回的是一列索引中的部分索引

            # 记录有效视角数（分母 |C_check|）
            valid_view_counts[valid_indices] += 1

            # 步骤2：从DPT深度图采样并对齐，得到目标深度z_ref [对应Algo 2 Line A.10]
            #  (1, 1, M, 2)
            grid = ndc[valid_indices].unsqueeze(0).unsqueeze(0)
            # 采样DPT相对深度
            d_raw = torch.nn.functional.grid_sample(
                dpt_depth_maps[idx].unsqueeze(0).unsqueeze(0),    # (1, 1, H, W)
                grid, align_corners=True, padding_mode="border"
            ).squeeze()
            # 得到（M,）

            # 应用对齐参数，将相对深度转换为度量深度 z_ref (z_render)
            a_c, b_c = alignment_params[idx]
            z_ref = a_c * d_raw + b_c

            # 步骤3：计算候选点的真实度量深度 z_metric
            p_view = cand_homo[valid_indices] @ cam.world_view_transform
            z_metric = p_view[:, 2]

            # 步骤4：计算深度误差 e = |z_metric - z_ref| [对应Algo 2 Line D.6]
            # diff只存储「当前视角有效点」的误差，索引范围是0~M - 1（局部索引），M是当前视角下的有效点数量
            diff = torch.abs(z_metric - z_ref)

            # 存入误差矩阵,一列
            error_matrix[valid_indices, idx] = diff

            # 更新最佳视角：选择误差最小的视角（用于后续几何纠偏）
            # better：布尔掩码 (M,)，标记当前视角误差是否小于该点已记录的最小视角误差（True=当前视角更优）
            better = diff < min_view_error[valid_indices]
            # update_idx：提取需要更新的有效点全局索引（筛选出当前视角更优的点）
            update_idx = valid_indices[better]
            # 更新这些点的最小视角误差为当前视角的误差值
            # 一维，N_rand
            min_view_error[update_idx] = diff[better]
            # 记录这些点的最佳视角索引为当前处理的视角idx
            best_view_idx[update_idx] = idx
            # 更新这些点的最佳参考深度为当前视角采样得到的z_ref
            best_z_ref[update_idx] = z_ref[better]


        # Phase 2: 计算动态阈值 eta_o [对应Eq.7]
        # eta_o = Q(Error_set, 0.9)：所有有效误差的90%分位数
        valid_mask_global = ~torch.isnan(error_matrix)  # 有效误差掩码，二维
        all_valid_errors = error_matrix[valid_mask_global]  # 所有有效误差值，一维

        if len(all_valid_errors) == 0:  # 无有效误差，直接返回
            return

        eta_o = torch.quantile(all_valid_errors, 0.7)
        # 最小值保护：避免阈值过小导致除零（工程鲁棒性）
        eta_o = torch.maximum(eta_o, torch.tensor(1e-8, device="cuda"))
        print(f" > 动态误差阈值 (eta_o): {eta_o:.4f}")

        # Phase 3: 应用阈值过滤 [对应Algo 2 Line D.7]
        # 仅保留误差小于eta_o的视角
        pass_mask = error_matrix < eta_o
        pass_mask = pass_mask & valid_mask_global  # 仅考虑有效视角
        pass_counts = pass_mask.sum(dim=1)  # 每个候选点通过的视角数

        # 最终筛选：通过视角数 >= min_views [对应Algo 2 Line D.8]
        final_mask = pass_counts >= min_views
        # 同时过滤掉无最佳视角的候选点
        # 理解：有了通过视角数不就有了最佳视角吗？为什么还要特地判断？
        final_idx = torch.where(final_mask & (best_view_idx != -1))[0]

        if len(final_idx) == 0:
            print(" > 无候选点通过一致性检查.")
            return

        # =================================================================
        # [Step E] 几何纠偏 (对应Algorithm 2 Part E)
        # 核心逻辑: 将候选点基于最佳视角的DPT深度回投影，校正到真实深度表面
        # 文档依据: "深度回投影校正...拉回 DPT 表面" [Section 4.3(ii)]
        # =================================================================
        # 筛选出最终有效的候选点索引
        idx_i_final = cand_i_cat[final_idx]
        idx_j_final = cand_j_cat[final_idx]
        final_xyz_raw = cand_mu_cat[final_idx]  # 纠偏前的原始候选点坐标

        # 预处理相机参数：相机中心和世界→相机变换矩阵
        all_cam_centers = torch.stack([c.camera_center for c in train_cameras])  # 所有相机中心 (N_cams, 3)
        all_w2c = torch.stack([c.world_view_transform for c in train_cameras])  # 所有相机的世界→视图矩阵 (N_cams, 4,4)

        # 获取最佳视角的参数 c* [对应Algo 2 Line E.1]
        best_cams_idx = best_view_idx[final_idx]  # 每个候选点的最佳视角索引
        centers = all_cam_centers[best_cams_idx]  # 最佳视角的相机中心 (N_final, 3)
        w2c_mats = all_w2c[best_cams_idx]  # 最佳视角的世界→视图矩阵 (N_final, 4,4)

        # 计算候选点在最佳视角下的当前深度 Z_current
        pts_homo = torch.cat([final_xyz_raw, torch.ones_like(final_xyz_raw[:, :1])], dim=-1)  # 齐次坐标
        # 转换到相机视图空间
        p_view = torch.bmm(pts_homo.unsqueeze(1), w2c_mats).squeeze(1)
        current_z = p_view[:, 2]  # 视图空间的z值（当前深度）

        # 获取目标深度 Z_target (z_render，来自DPT的对齐深度)
        target_z = best_z_ref[final_idx]

        # 深度回投影纠偏 [对应Algo 2 Line E.3]
        # 核心公式：new_xyz = cam_center + (raw_xyz - cam_center) * (target_z / current_z)
        # 理解，怎么算的？这里的工程约束真的可靠吗？-------------------------------------------------------
        ratio = target_z / (current_z + 1e-8)  # 深度缩放比例
        ratio = torch.clamp(ratio, 0.8, 1.2)  # 工程约束：缩放比例限制在[0.5,2.0]，避免过度纠偏
        final_xyz = centers + (final_xyz_raw - centers) * ratio.unsqueeze(1)  # 纠偏后的最终坐标


        # =================================================================
        # [Step F] 可信度门控初始化 (对应Algorithm 2 Part F)
        # 核心公式: q_new = clip(cnt / |C_check|) * max(0, 1 - median(E)/eta_o)
        # 作用：基于一致性和误差计算候选点的可信度，初始化不透明度和缩放
        # =================================================================
        # 初始化新点的特征（取触发点和邻居点的均值）
        f_dc = 0.5 * (self._features_dc[idx_i_final] + self._features_dc[idx_j_final])  # 直流特征
        f_rest = 0.5 * (self._features_rest[idx_i_final] + self._features_rest[idx_j_final])  # 高频特征
        rot = self._rotation[idx_i_final]  # 旋转参数（继承触发点）


        # Term 1: 通过率 Ratio [对应Algo 2 Line F.1 Part 1]
        # cnt / |C_check|，限制在[0,1]（避免分母为0，取最大值1）
        # valid_view_counts[final_idx]这些候选点的有效视角数量（候选点在所在的视角的视锥内则称该视角为该候选点的有效视角）
        denom = torch.maximum(valid_view_counts[final_idx], torch.tensor(1, device="cuda"))
        term_ratio = torch.clamp(pass_counts[final_idx].float() / denom.float(), 0.0, 1.0)

        # Term 2: 误差门控（严格中位数）[对应Algo 2 Line F.1 Part 2]
        passed_errors_subset = error_matrix[final_idx]  # 最终候选点的误差矩阵
        mask_subset = pass_mask[final_idx]  # 通过阈值的误差掩码
        passed_errors_subset[~mask_subset] = float('nan')  # 无效误差标记为nan

        # 计算每个候选点通过阈值的误差的中位数 median(E)
        median_vals = torch.nanmedian(passed_errors_subset, dim=1).values
        # 误差门控项：1 - median(E)/eta_o，限制最小值为0
        term_error = torch.clamp(1.0 - median_vals / (eta_o + 1e-8), min=0.0)

        # 计算最终可信度 q_new
        q_new = (term_ratio * term_error).unsqueeze(1)

        # 门控应用：初始化不透明度和缩放 [对应Algo 2 Line F.3]
        opac = self.get_opacity[idx_i_final]  # 触发点的原始不透明度
        scale = self.get_scaling[idx_i_final]  # 触发点的原始缩放

        # 计算新点的不透明度（应用缩放和可信度门控）
        final_opac = self.inverse_opacity_activation(opac * opacity_scale * q_new)
        # 计算新点的缩放（尺寸收缩 + 可信度门控）
        scale_mod = (1.0 / size_shrink) * (0.5 + 0.5 * q_new)
        final_scale = self.scaling_inverse_activation(scale * scale_mod)

        # 将新点添加到高斯模型中
        self.densification_postfix(final_xyz, f_dc, f_rest, final_opac, final_scale, rot)
        print(f"[SaGPD] 预致密化完成. 新增 {len(final_xyz)} 个高斯点.")