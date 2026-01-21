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
from glob import glob
from argparse import ArgumentParser
import re

# 解析命令行参数
parser = ArgumentParser("场景指标汇总工具")
parser.add_argument("--path", "-s", required=True, type=str, help="实验结果根目录，例如 output/llff_sagpd_3view")
parser.add_argument("--iteration", "-i", required=False, type=str, default='10000', help="评估的迭代次数")
args = parser.parse_args()

# 初始化统计变量
PSNR_total = 0
SSIM_total = 0
LPIPS_total = 0
Added_total = 0
results = []  # 用于存储每个场景的详细数据

metric_path = os.path.join(args.path, 'metrics_mean.txt')
if os.path.exists(metric_path):
    os.remove(metric_path)

# 获取所有子目录并排序，确保表格顺序一致
dir_lst = sorted(glob(args.path + '/*'))
valid_count = 0

for d in dir_lst:
    # 确保是目录且包含目标指标文件
    if not os.path.isdir(d):
        continue

    file_path = os.path.join(d, 'metrics_{}.txt'.format(args.iteration))
    if not os.path.exists(file_path):
        continue

    scene_name = os.path.basename(d)
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.splitlines()

            # 使用正则提取前三行基础指标
            psnr = float(re.findall(r"[-+]?\d*\.\d+|\d+", lines[0])[0])
            ssim = float(re.findall(r"[-+]?\d*\.\d+|\d+", lines[1])[0])
            lpips = float(re.findall(r"[-+]?\d*\.\d+|\d+", lines[2])[0])

            # 新增：提取新增点数 (寻找 Added_Points 关键字)
            added_pts = 0
            added_match = re.search(r"Added_Points\s*:\s*(\d+)", content)
            if added_match:
                added_pts = int(added_match.group(1))

            results.append({
                "scene": scene_name,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "added": added_pts
            })

            PSNR_total += psnr
            SSIM_total += ssim
            LPIPS_total += lpips
            Added_total += added_pts
            valid_count += 1
    except Exception as e:
        print(f"读取场景 {scene_name} 失败: {e}")

# 输出结果汇总
if valid_count > 0:
    PSNR_avg = PSNR_total / valid_count
    SSIM_avg = SSIM_total / valid_count
    LPIPS_avg = LPIPS_total / valid_count
    Added_avg = Added_total / valid_count

    # --- 打印格式化表格 ---
    # 调整了宽度以容纳 Added Pts 列
    header = f"{'场景 (Scene)':<15} | {'Added Pts':<10} | {'PSNR (↑)':<10} | {'SSIM (↑)':<10} | {'LPIPS (↓)':<10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        print(f"{r['scene']:<15} | {r['added']:<10} | {r['psnr']:<10.4f} | {r['ssim']:<10.4f} | {r['lpips']:<10.4f}")

    print("-" * len(header))
    print(f"{'均值 (Mean)':<15} | {int(Added_avg):<10} | {PSNR_avg:<10.4f} | {SSIM_avg:<10.4f} | {LPIPS_avg:<10.4f}")
    print("=" * len(header) + "\n")

    # --- 写入原有的 metrics_mean.txt 保持逻辑兼容 ---
    with open(metric_path, 'w') as f:
        f.write('PSNR : {:>12.7f}\n'.format(PSNR_avg))
        f.write('SSIM : {:>12.7f}\n'.format(SSIM_avg))
        f.write('LPIPS : {:>12.7f}\n'.format(LPIPS_avg))
        f.write('Mean_Added_Points : {:>12.1f}\n'.format(Added_avg))

    print(f"汇总指标已保存至: {metric_path}")
else:
    print(f"在路径 {args.path} 下未找到迭代次数为 {args.iteration} 的有效指标文件。")