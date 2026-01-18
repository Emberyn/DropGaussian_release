// /*
//  * Copyright (C) 2023, Inria
//  * GRAPHDECO research group, https://team.inria.fr/graphdeco
//  * All rights reserved.
//  *
//  * This software is free for non-commercial, research and evaluation use
//  * under the terms of the LICENSE.md file.
//  *
//  * For inquiries contact  george.drettakis@inria.fr
//  */
//
// #include "spatial.h"
// #include "simple_knn.h"
//
// torch::Tensor
// distCUDA2(const torch::Tensor& points)
// {
//   const int P = points.size(0);
//
//   auto float_opts = points.options().dtype(torch::kFloat32);
//   torch::Tensor means = torch::full({P}, 0.0, float_opts);
//
//   SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());
//
//   return means;
// }

/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "simple_knn.h"
// 输入参数 points 是一个形状为 (P, 3) 的 torch::Tensor，其中 P 是点的数量，且每个点包含 (x, y, z) 三维坐标。
std::tuple<torch::Tensor, torch::Tensor> distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  auto int_opts = points.options().dtype(torch::kInt32);
  // 创建了一个大小为 (P, 3) 的张量，计划用来存储每个点的 3 个最近邻点的索引
  torch::Tensor nearestIndices = torch::empty({P, 3}, int_opts).set_requires_grad(false);
  // 一个浮点型张量，形状为 (P)，计划用来存储每个点到其 3 个最近邻点的平均距离。
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  // 调用SimpleKNN::knn 是一个 CUDA 实现的 K 近邻算法，负责在 GPU 上计算每个点的最近邻
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>(), nearestIndices.contiguous().data<int32_t>());

  return std::make_tuple(means, nearestIndices);
}