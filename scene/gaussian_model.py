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
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
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




    def apply_SaGPD(self, scene_extent, K=8, tau_s_quantile=0.7, gamma_o=0.3, delta=1.5, eta_t=0.1):
        """
        SaGPD A++ 去噪版：
        1. 表面可靠性过滤：通过 PCA 特征值比例，剔除在“噪声点团”中的插值。
        2. 唯一边去重 + 硬预算截断。
        3. 仅在真正具有“表面特征”的区域补点，提升稀疏视图的泛化性。
        """
        torch.cuda.empty_cache()
        xyz = self.get_xyz
        N_base = xyz.shape[0]

        MAX_TOTAL_POINTS = 60000
        budget = MAX_TOTAL_POINTS - N_base
        if budget <= 0: return

        # 邻域分析
        dist_matrix = torch.cdist(xyz, xyz)
        dist_k, idx_k = torch.topk(dist_matrix, k=K + 1, largest=False)
        dist_k, idx_k = dist_k[:, 1:], idx_k[:, 1:]
        D_i = dist_k.mean(dim=1)
        d_t = torch.median(D_i)

        D_hat = (D_i - D_i.min()) / (D_i.max() - D_i.min() + 1e-12)
        tau_s = torch.quantile(D_hat, tau_s_quantile)
        sparse_mask = D_hat > tau_s

        candidates = []
        sparse_indices = torch.where(sparse_mask)[0]

        for i in sparse_indices:
            # --- 关键改进：表面可靠性检查 ---
            neighbors_xyz = xyz[idx_k[i]]
            centered = neighbors_xyz - neighbors_xyz.mean(dim=0)
            cov = (centered.T @ centered) / K
            # u_eig 是特征值（从小到大），v_eig 是特征向量
            u_eig, v_eig = torch.linalg.eigh(cov)

            # 计算平面度 (Planarity Score): 1 - (最小特征值 / 平均特征值)
            # 如果 planarity 接近 1，说明非常平；如果接近 0，说明是球状噪声
            planarity = 1.0 - (u_eig[0] / (u_eig.mean() + 1e-12))

            # 只有当邻域看起来像一个平面时 (阈值 0.8)，才允许从这个点出发寻找长边
            if planarity < 0.8:
                continue

            u_base, v_base = v_eig[:, 1], v_eig[:, 2]

            for j_idx in range(K):
                j = idx_k[i, j_idx]
                if i < j:
                    L_ij = dist_matrix[i, j].item()
                    if L_ij > d_t.item() * 1.2:
                        candidates.append({
                            'length': L_ij, 'i': i.item(), 'j': j.item(),
                            'u': u_base, 'v': v_base
                        })

        # 按长度排序
        candidates = sorted(candidates, key=lambda x: x['length'], reverse=True)

        new_xyz_list, new_features_dc_list, new_features_rest_list = [], [], []
        new_scaling_list, new_rotation_list, new_opacity_list = [], [], []

        added_count = 0
        for edge in candidates:
            if added_count >= budget: break
            i, j, L_ij = edge['i'], edge['j'], edge['length']
            m_ij = int(torch.clamp(torch.tensor(L_ij / d_t.item() - 1), min=1, max=3))

            for r in range(1, m_ij + 1):
                if added_count >= budget: break
                t_r = r / (m_ij + 1)
                mu_lerp = (1 - t_r) * xyz[i] + t_r * xyz[j]

                # 使用预存的 PCA 基向量
                alpha = torch.randn(2, device="cuda") * (eta_t * d_t)
                mu_new = mu_lerp + alpha[0] * edge['u'] + alpha[1] * edge['v']

                new_xyz_list.append(mu_new)
                new_features_dc_list.append((1 - t_r) * self._features_dc[i] + t_r * self._features_dc[j])
                new_features_rest_list.append((1 - t_r) * self._features_rest[i] + t_r * self._features_rest[j])
                new_opacity_list.append(self.inverse_opacity_activation(self.get_opacity[i] * gamma_o))
                scale_new = self._scaling[i] - torch.log(torch.tensor(delta * (m_ij + 1) ** 0.5, device="cuda"))
                new_scaling_list.append(scale_new)
                new_rotation_list.append(self._rotation[i])
                added_count += 1

        if added_count > 0:
            self.densification_postfix(
                torch.stack(new_xyz_list), torch.stack(new_features_dc_list),
                torch.stack(new_features_rest_list), torch.stack(new_opacity_list),
                torch.stack(new_scaling_list), torch.stack(new_rotation_list)
            )
            print(f"[SaGPD] 过滤后精准增密完成！新增: {len(new_xyz_list)}, 最终点数: {self.get_xyz.shape[0]}")

        torch.cuda.empty_cache()