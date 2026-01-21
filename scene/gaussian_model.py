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




    @torch.no_grad()  # 禁用梯度计算：该函数仅为推理/增密流程，无需反向传播
    # 优势：大幅节省GPU显存，避免不必要的梯度计算开销
    def apply_SaGPD(self, scene, pipe, background,
                    knn_neighbors=8,
                    sparsity_threshold=0.7,
                    opacity_scale=0.3,
                    size_shrink=1.5,
                    perturb_strength=0.1,
                    min_views=2,
                    depth_error=0.01):
        """
        SaGPD A++ (修复版): Surface-aware Gaussian Point Densification 表面感知高斯点增密算法
        参数详解：
            self: 高斯点云类实例，包含点云核心属性（xyz坐标、特征、透明度、缩放、旋转等）
            scene: 场景对象，提供训练相机列表、场景范围等核心信息
            pipe: 渲染管道（预留参数，当前版本未实际使用）
            background: 背景色（预留参数，当前版本未实际使用）
            knn_neighbors (int): K近邻数量，用于局部邻域几何特征分析（默认8个）
            sparsity_threshold (float): 稀疏区域筛选分位数阈值（0-1），值越大筛选的稀疏点越少
            opacity_scale (float): 新增点透明度衰减系数（0-1），控制新增点透明度强度
            size_shrink (float): 新增点缩放衰减系数，边越长/插入点数越多，缩放越小
            perturb_strength (float): 平面内随机扰动强度，模拟表面微小起伏
            min_views (int): 多视图一致性校验最小通过视角数，低于该值的候选点会被过滤
            depth_error (float): 深度一致性校验的相对误差基准
        """
        # 导入必要依赖库
        import math  # 用于相机视角相关的三角函数计算（tan(fov/2)）
        # 导入高斯光栅化核心模块：实现高斯点的深度图渲染
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        from tqdm import tqdm  # 导入进度条库，可视化深度图渲染进度

        # 第一步：清空CUDA缓存，释放之前未回收的显存，避免增密流程显存不足
        torch.cuda.empty_cache()

        # 获取当前高斯点云的三维坐标张量 (N, 3)，N为原始点云的点数
        xyz = self.get_xyz
        # 记录原始点云的点数，用于计算增密预算
        N_base = xyz.shape[0]

        # ===================== 增密预算控制 =====================
        # 设定高斯点云总数的硬上限（防止增密后点数过多导致显存/计算量爆炸）
        MAX_TOTAL_POINTS = 100000
        # 计算可新增的点数 = 总点数上限 - 原始点数
        budget = MAX_TOTAL_POINTS - N_base
        # 若预算≤0（原始点数已达上限），直接返回，不执行后续增密流程
        if budget <= 0:
            print("[SaGPD] 当前点数已达上限，跳过增密。")
            return

        # ===================== 阶段 1: 预渲染所有训练视角的深度图 =====================
        # 核心目的：生成每个训练相机视角下的深度参考图，为后续多视图一致性校验提供基准
        print(f"[SaGPD] 正在生成深度图参考...")
        # 获取场景中所有训练视角的相机对象列表（每个相机包含内参、外参、变换矩阵等）
        train_cameras = scene.getTrainCameras()
        # 初始化列表，存储每个训练相机视角下的深度图（后续移到CPU节省显存）
        ref_depths = []

        # 提取当前高斯点云的核心属性，用于深度图渲染
        means3D = self.get_xyz  # 高斯点三维坐标 (N, 3)
        opacity = self.get_opacity  # 高斯点透明度 (N, 1)
        scales = self.get_scaling  # 高斯点缩放因子 (N, 3)
        rotations = self.get_rotation  # 高斯点旋转四元数 (N, 4)

        # 预计算齐次坐标：将三维坐标 (x,y,z) 扩展为四维齐次坐标 (x,y,z,1) (N, 4)
        # 优化点：提前计算一次，避免在相机循环内重复计算，提升执行效率
        means3D_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)

        # 遍历所有训练相机，渲染每个视角的深度图（添加tqdm进度条，可视化渲染进度）
        for cam in tqdm(train_cameras, desc="Rendering Depths"):
            # 计算相机水平/垂直方向的tan(fov/2)，是光栅化器的必要参数
            tanfovx = math.tan(cam.FoVx * 0.5)
            tanfovy = math.tan(cam.FoVy * 0.5)

            # 配置高斯光栅化器的参数（用于深度图渲染）
            raster_settings = GaussianRasterizationSettings(
                image_height=int(cam.image_height),  # 渲染深度图的高度（与相机分辨率一致）
                image_width=int(cam.image_width),  # 渲染深度图的宽度（与相机分辨率一致）
                tanfovx=tanfovx,  # 水平方向tan(fov/2)
                tanfovy=tanfovy,  # 垂直方向tan(fov/2)
                bg=torch.zeros(3, device="cuda"),  # 背景色设为黑色（深度图渲染无需背景色）
                scale_modifier=1.0,  # 尺度修正因子（默认1.0，无膨胀/收缩）
                viewmatrix=cam.world_view_transform,  # 相机视角变换矩阵（世界→相机坐标系）
                projmatrix=cam.full_proj_transform,  # 相机投影变换矩阵（相机→NDC坐标系）
                sh_degree=0,  # SH阶数设为0（仅渲染深度，无需颜色）
                campos=cam.camera_center,  # 相机中心坐标（世界坐标系）
                prefiltered=False,  # 禁用预过滤（深度渲染无需抗锯齿）
                debug=False  # 禁用调试模式
            )

            # 根据配置初始化高斯光栅化器
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            # === [关键修复] ===
            # 计算结果：means3D_cam (N,4)，第3列（索引2）为相机坐标系下的深度（Z轴值）
            means3D_cam = (means3D_homo @ cam.world_view_transform)
            depths = means3D_cam[:, 2:3]  # 提取相机坐标系下的Z轴值作为深度 (N,1)

            # 初始化2D坐标张量（光栅化器接口要求，深度渲染无需实际2D坐标，故设为全0）
            means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=False, device="cuda")

            # 执行高斯光栅化，渲染当前相机视角的深度图
            depth_image, _ = rasterizer(
                means3D=means3D,  # 高斯点三维坐标
                means2D=means2D,  # 高斯点2D坐标（全0）
                shs=None,  # 无SH系数（仅渲染深度，无需颜色）
                colors_precomp=depths.repeat(1, 3),  # 深度值转为3通道（适配光栅化器颜色输入）
                opacities=opacity,  # 高斯点透明度（控制深度图融合）
                scales=scales,  # 高斯点缩放因子
                rotations=rotations,  # 高斯点旋转四元数
                cov3D_precomp=None  # 无需预计算3D协方差（光栅化器自动计算）
            )

            # === [显存优化] ===
            # 优化逻辑：将渲染完成的深度图从GPU移到CPU存储，释放GPU显存
            # 原因：深度图仅在校验时临时取回GPU，全程存在GPU会导致显存堆积（多相机视角下易OOM）
            # 细节：仅保留深度图单通道（[0:1]），进一步减少内存占用,(1, 512, 512)
            ref_depths.append(depth_image[0:1].cpu())

        # 渲染完所有深度图后，再次清空CUDA缓存，释放渲染过程中临时占用的显存
        torch.cuda.empty_cache()

        # ===================== 阶段 2: 邻域分析（高效显存版） =====================
        # 核心目的：只计算密度指标 D_i 和阈值 d_t，不再强行保存庞大的近邻表
        print("[SaGPD] 分析几何特征 (阶段2: 密度筛选)...")

        # 尝试使用 simple_knn 快速计算密度
        try:
            from simple_knn._C import distCUDA2
            mean_dist_sq, _ = distCUDA2(xyz)
            D_i = torch.sqrt(mean_dist_sq)
        except ImportError:
            # 回退方案：分块计算密度
            print("[SaGPD] 未找到 simple_knn，使用分块计算密度...")
            D_i = torch.zeros(N_base, device="cuda")
            for start in range(0, N_base, 2048):
                end = min(start + 2048, N_base)
                # 只计算距离均值，不保存索引，用完即弃
                dist_batch = torch.cdist(xyz[start:end], xyz)
                vals, _ = torch.topk(dist_batch, k=knn_neighbors + 1, largest=False)
                D_i[start:end] = vals[:, 1:].mean(dim=1)

        # 计算全局密度阈值
        d_t = torch.median(D_i)

        # 筛选稀疏点
        D_hat = (D_i - D_i.min()) / (D_i.max() - D_i.min() + 1e-12)
        tau_s = torch.quantile(D_hat, sparsity_threshold)
        sparse_indices = torch.where(D_hat > tau_s)[0]

        # ===================== 阶段 3: 候选点生成（自包含计算版） =====================
        # 核心修复：在此处即时计算近邻，不再依赖不存在的全局 idx_k
        print(f"[SaGPD] 正在批量分析几何特征 (需处理稀疏点数: {len(sparse_indices)})...")
        pre_candidates = []

        CHUNK_SIZE = 2048  # 批处理大小

        for start in range(0, len(sparse_indices), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(sparse_indices))
            batch_indices = sparse_indices[start:end]  # (B,)

            # 获取当前批次稀疏点的坐标
            batch_xyz = xyz[batch_indices]  # (B, 3)

            # --- 核心修复开始：即时计算当前批次的 KNN ---
            # 计算 (B, N) 的距离矩阵，而不是 (N, N)，显存安全
            dist_batch = torch.cdist(batch_xyz, xyz)

            # 获取 TopK 近邻 (B, K+1)
            # 这里的 batch_idx_k 对应的是全图中的索引
            batch_dist_k, batch_idx_k = torch.topk(dist_batch, k=knn_neighbors + 1, largest=False)

            # 剔除自身 (第一列)
            curr_dists = batch_dist_k[:, 1:]  # (B, K) -> 对应之前的 dist_k
            curr_neighbors_idx = batch_idx_k[:, 1:]  # (B, K) -> 对应之前的 idx_k
            # --- 核心修复结束 ---

            # 1. 提取邻域坐标进行几何分析 (B, K, 3)
            # 利用刚刚算出来的索引提取坐标
            # 注意：idx_k 是全局索引，所以要从全局 xyz 中取
            neighbors_xyz = xyz[curr_neighbors_idx]

            # 2. 批量计算协方差 (B, 3, 3)
            centered = neighbors_xyz - neighbors_xyz.mean(dim=1, keepdim=True)
            covs = torch.matmul(centered.transpose(1, 2), centered) / knn_neighbors

            # 3. 批量特征值分解
            eigvals, eigvecs = torch.linalg.eigh(covs)

            # 4. 平面度筛选
            planarities = 1.0 - (eigvals[:, 0] / (eigvals.mean(dim=1) + 1e-12))
            valid_planarity_mask = planarities > 0.8

            # 5. 提取基向量
            u_bases = eigvecs[:, :, 1]
            v_bases = eigvecs[:, :, 2]

            # 6. 处理长边 (在有效平面内)
            valid_batch_subs = torch.where(valid_planarity_mask)[0]

            for sub_idx in valid_batch_subs:
                i = batch_indices[sub_idx]  # 全局索引 i

                # 获取当前点的近邻列表 (从即时计算的结果中取)
                this_neighbors = curr_neighbors_idx[sub_idx]
                this_dists = curr_dists[sub_idx]

                for k_idx in range(knn_neighbors):
                    j = this_neighbors[k_idx]  # 全局索引 j

                    # 逻辑保持不变
                    if i < j:
                        L_ij = this_dists[k_idx].item()
                        if L_ij > d_t.item() * 1.2:
                            pre_candidates.append({
                                'i': i.item(),
                                'j': j.item(),
                                'L': L_ij,
                                'u': u_bases[sub_idx],
                                'v': v_bases[sub_idx]
                            })

        # ===================== 阶段 3.5: 排序与预算 =====================
        if len(pre_candidates) > 0:
            pre_candidates = sorted(pre_candidates, key=lambda x: x['L'], reverse=True)
            pre_candidates = pre_candidates[:budget * 2]
            print(f"[SaGPD] 几何分析完成。筛选出 {len(pre_candidates)} 条高质量候选边。")
        else:
            print("[SaGPD] 未找到符合要求的候选区域。")
            return


        # ===================== 阶段 4: 多视图一致性校验（核心鲁棒性保障） =====================
        # 核心目的：校验候选点的深度是否符合多视角物理约束，过滤错误的增密点
        tmp_xyz_list = []  # 存储所有候选增密点的坐标
        meta_info = []  # 存储候选点的元信息（用于后续生成属性）

        # 遍历候选长边，生成初始候选增密点
        for edge in pre_candidates:
            # 计算当前边需要插入的点数 m_ij：
            # 公式：L_ij/d_t - 1 → 边越长，插入点数越多；
            # clamp限制：最少插入1个点，最多插入3个点（避免单条边插入过多点）
            m_ij = int(torch.clamp(torch.tensor(edge['L'] / d_t.item() - 1), min=1, max=3))
            # 遍历每个插入位置，生成候选点
            for r in range(1, m_ij + 1):
                # 插值系数 t_r：在(0,1)区间均匀分布（避免插入到边的两个端点）
                t_r = r / (m_ij + 1)
                # 计算i-j边上的线性插值基准点（无偏移的原始插值点）
                mu_lerp = (1 - t_r) * xyz[edge['i']] + t_r * xyz[edge['j']]
                # 生成平面内的随机偏移：模拟表面微小起伏，提升增密后表面的真实感
                # alpha：正态分布随机数，尺度为 perturb_strength*d_t（控制偏移幅度）
                alpha = torch.randn(2, device="cuda") * (perturb_strength * d_t)
                # 最终候选点坐标 = 插值基准点 + 切向方向的随机偏移
                mu_new = mu_lerp + alpha[0] * edge['u'] + alpha[1] * edge['v']
                # 保存候选点坐标和元信息
                tmp_xyz_list.append(mu_new)
                meta_info.append({'i': edge['i'], 'j': edge['j'], 't_r': t_r, 'm_ij': m_ij})

        # 若无候选点，直接返回（无需后续校验）
        if not tmp_xyz_list:
            print("[SaGPD] 未找到符合条件的候选点。")
            return

        # 将候选点坐标堆叠为张量 (M, 3)，M为候选点总数
        all_mu_new = torch.stack(tmp_xyz_list)
        # 初始化一致性计数张量 (M,)：记录每个候选点通过的视角数（初始为0）
        consistency_count = torch.zeros(all_mu_new.shape[0], device="cuda")

        print("[SaGPD] 执行多视图一致性校验...")
        # 将候选点转换为齐次坐标 (M, 4)，方便后续相机视角/投影变换
        p_homo = torch.cat([all_mu_new, torch.ones(all_mu_new.shape[0], 1, device="cuda")], dim=-1)

        # 遍历所有训练相机，校验候选点的深度一致性
        for idx, cam in enumerate(train_cameras):
            # 将候选点齐次坐标转换到相机NDC坐标系（归一化设备坐标）   [X, Y, Z, W]
            p_cam = (p_homo @ cam.full_proj_transform)
            z_new = p_cam[:, 2]  # 候选点在相机坐标系的深度（Z轴值）
            ndc_x = p_cam[:, 0] / p_cam[:, 3]  # NDC x坐标（范围[-1,1]）
            ndc_y = p_cam[:, 1] / p_cam[:, 3]  # NDC y坐标（范围[-1,1]）

            # 筛选出在相机视锥内的候选点：NDC坐标在[-1,1]且深度>0（有效可见点）
            mask = (ndc_x >= -1) & (ndc_x <= 1) & (ndc_y >= -1) & (ndc_y <= 1) & (z_new > 0)

            if mask.any():  # 若有视锥内的候选点，执行深度一致性校验
                # 构造采样网格：适配grid_sample的输入格式 (1, 1, N_valid, 2)
                grid = torch.stack([ndc_x[mask], ndc_y[mask]], dim=-1).unsqueeze(0).unsqueeze(0)

                # 从CPU取回当前视角的深度图到GPU，用于深度采样
                curr_depth_map = ref_depths[idx].to("cuda")

                # 从参考深度图中采样候选点位置的深度值（双线性插值）
                z_ref = torch.nn.functional.grid_sample(
                    curr_depth_map.unsqueeze(0),  # 扩展为 (1,1,H,W) 格式
                    grid,  # 采样网格（候选点的NDC坐标）
                    align_corners=True,  # 对齐角落像素（保证采样精度）
                    padding_mode="border"  # 边界填充模式（超出范围取边界值）
                ).squeeze()  # 压缩维度，得到 (N_valid,) 的深度值

                # 计算候选点深度与参考深度的绝对误差
                dist_diff = torch.abs(z_new[mask] - z_ref)
                # 判断深度一致性：误差小于（相对误差+绝对误差）的候选点为有效点
                # 公式：z_ref*depth_error（相对误差） + 0.05（绝对误差），兼顾不同深度的误差容忍度
                consistent_mask = dist_diff < (z_ref * depth_error + 0.05)
                # 更新一致性计数：通过校验的候选点，计数+1
                consistency_count[torch.where(mask)[0][consistent_mask]] += 1

        # ===================== 阶段 5: 筛选有效点并添加到点云 =====================
        # 筛选最终有效点：一致性计数≥最小通过视角数，且不超过增密预算
        valid_mask = consistency_count >= min_views
        final_indices = torch.where(valid_mask)[0][:budget]

        # 初始化新增点的各类属性列表
        new_xyz, new_dc, new_rest, new_opac, new_scale, new_rot = [], [], [], [], [], []

        # 遍历最终有效点，生成新增点的完整属性
        for idx in final_indices:
            m = meta_info[idx]
            i, j, t_r = m['i'], m['j'], m['t_r']
            # 新增点坐标（经过一致性校验的有效坐标）
            new_xyz.append(all_mu_new[idx])
            # 特征DC分量：线性插值继承边的两个端点的特征（保证特征连续性）
            new_dc.append((1 - t_r) * self._features_dc[i] + t_r * self._features_dc[j])
            # 特征剩余分量：线性插值继承边的两个端点的特征
            new_rest.append((1 - t_r) * self._features_rest[i] + t_r * self._features_rest[j])
            # 透明度：继承起点i的透明度并乘以衰减系数，通过反激活转回原始空间
            new_opac.append(self.inverse_opacity_activation(self.get_opacity[i] * opacity_scale))
            # 缩放：继承起点i的缩放并按边长度/插入点数衰减（边越长/点数越多，缩放越小）
            s_base = self._scaling[i]
            scale_val = s_base - torch.log(torch.tensor(size_shrink * (m['m_ij'] + 1) ** 0.5, device="cuda"))
            new_scale.append(scale_val)
            # 旋转：直接继承起点i的旋转（保证表面朝向的一致性）
            new_rot.append(self._rotation[i])

        # 若有新增点，调用后处理函数将新增点合并到原始点云
        if len(new_xyz) > 0:
            self.densification_postfix(
                torch.stack(new_xyz),  # 新增点坐标 (M, 3)
                torch.stack(new_dc),  # 新增点特征DC (M, C)
                torch.stack(new_rest),  # 新增点特征剩余 (M, C*L)
                torch.stack(new_opac),  # 新增点透明度 (M, 1)
                torch.stack(new_scale),  # 新增点缩放 (M, 3)
                torch.stack(new_rot)  # 新增点旋转 (M, 4)
            )
            # 打印增密结果日志
            print(f"[SaGPD] 预密集化完成！新增点数: {len(new_xyz)}, 总点数: {self.get_xyz.shape[0]}")
        else:
            # 无有效点通过校验，打印提示
            print("[SaGPD] 校验完成，没有点通过一致性测试。")

        # 最终清空CUDA缓存，释放所有临时变量占用的显存
        torch.cuda.empty_cache()